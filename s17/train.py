from model import build_transformer
from dataset import BillingualDataset, casual_mask
from config_file import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
torch.cuda.amp.autocast(enabled = True)

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import OneCycleLR
import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12240"
config = get_config()
import glob

def get_last_saved_model(config):
    model_directory = config['model_folder']
    model_pattern = config['model_basename'] + "*.pt"
    model_files = glob.glob(os.path.join(model_directory, model_pattern))
    model_files.sort(key=os.path.getmtime, reverse=True)

    if model_files:
        print(model_files)
        return model_files[0]
    else:
        return None
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    
    
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    encoder_output = model.encode(source, source_mask)
    #Initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source_mask).fill_(next_word.item()).to(device)
            ],
            dim =  1
        )
        
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)


def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, writer, global_step):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
    
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
        
    with torch.no_grad():
        for batch in val_dataloader:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            assert encoder_input.size(0)==1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
    """        
            print("SOURCE", source_text)
            print("TARGET", target_text)
            print("PREDICTED", model_out_text)
            
    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()
        
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()
        
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
        
     """   

def get_all_sentenses(ds, lang):
    for item in ds:
        yield item['translation'][lang]
        
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace() 
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentenses(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split = 'train')  
    
    src_lang = config["lang_src"]
    tgt_lang = config["lang_tgt"]
    seq_len = config["seq_len"]
    
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, src_lang)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, tgt_lang)
    
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len)
    val_ds = BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len)
    
    # max_len_src = 0
    # max_len_tgt = 0
    
    # for item in ds_raw:
    #     src_ids = tokenizer_src.encode(item['translation'][src_lang]).ids
    #     tgt_ids = tokenizer_tgt.encode(item['translation'][tgt_lang]).ids
    #     max_len_src = max(max_len_src, len(src_ids))
    #     max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    # print(f"Max length of the source sentence : {max_len_src}")
    # print(f"Max length of the source target : {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size = config["batch_size"], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config["seq_len"], config["seq_len"], d_model=config['d_model'])
    return model


# def train_model(config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device : {device}")
    
#     Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
#     train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
#     model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
#     #Tensorboard
#     writer = SummaryWriter(config["experiment_name"])
    
#     #Adam is used to train each feature with a different learning rate. 
#     #If some feature is appearing less, adam takes care of it
#     optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], eps = 1e-9)
    
#     initial_epoch = 0
#     global_step = 0
    
#     if config["preload"]:
#         model_filename = get_weights_file_path(config, config["preload"])
#         print("Preloading model {model_filename}")
#         state = torch.load(model_filename)
#         model.load_state_dict(state["model_state_dict"])
#         initial_epoch = state["epoch"] + 1
#         optimizer.load_state_dict(state["optimizer_state_dict"])
#         global_step = state["global_step"]
#         print("preloaded")
        
#     loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)
    
#     for epoch in range(initial_epoch, config["num_epochs"]):
#         torch.cuda.empty_cache()
#         print(epoch)
#         model.train()
#         batch_iterator = tqdm(train_dataloader, desc = f"Processing Epoch {epoch:02d}")
        
#         for batch in batch_iterator:
#             encoder_input = batch["encoder_input"].to(device, non_blocking=True)
#             decoder_input = batch["decoder_input"].to(device, non_blocking=True)
#             encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)
#             decoder_mask = batch["decoder_mask"].to(device, non_blocking=True)
            
#             encoder_output = model.encode(encoder_input, encoder_mask)
#             decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
#             proj_output = model.project(decoder_output)
            
#             label = batch["label"].to(device, non_blocking=True)
            
#             #Compute loss using cross entropy
#             tgt_vocab_size = tokenizer_tgt.get_vocab_size()
#             loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
#             batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

#             #Log the loss
#             writer.add_scalar('train_loss', loss.item(), global_step)
#             writer.flush()
            
#             #Backpropogate loss
#             loss.backward()
            
#             #Update weights
#             optimizer.step()
#             optimizer.zero_grad(set_to_none=True)
#             global_step+=1
            
#         #run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, writer, global_step)
        
        
#         model_filename = get_weights_file_path(config, f"{epoch:02d}")
#         torch.save(
#             {
#                 "epoch": epoch,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "global_step": global_step
#             },
#             model_filename
#         )
def train_model(config):
    global enable_amp
    enable_amp = config['enable_amp']
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"👀Using device: {device}")

    # Weights folder
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config=config,
                      src_vocab_size=tokenizer_src.get_vocab_size(),
                      tgt_vocab_size=tokenizer_tgt.get_vocab_size()
                      ).to(device)

    # Summary Writer for Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1.0e-9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)

    # optimizer = Lion(model.parameters(), lr=config['lr']/3, weight_decay=1e-2)

    scheduler = OneCycleLR(optimizer=optimizer,
                           max_lr=config['max_lr'],
                           steps_per_epoch=len(train_dataloader),
                           epochs=config['num_epochs'],
                           pct_start=config['pct_start'],
                           anneal_strategy=config['anneal_strategy'],
                           div_factor=config['initial_div_factor'],
                           final_div_factor=config['final_div_factor'],
                           three_phase=config['three_phase'])
    # scheduler = None

    # if a model is specified for preload before training then load it
    run_val = False
    initial_epoch = 0
    global_step = 0
    lrs = []

    if config['preload']:
        model_filename = get_last_saved_model(config)

        print(f"Preloading model {model_filename}")

        state = torch.load(model_filename)

        model.load_state_dict(state_dict=state['model_state_dict'])

        initial_epoch = state['epoch'] + 1

        optimizer.load_state_dict(state['optimizer_state_dict'])

        global_step = state['global_step']

        scaler.load_state_dict(state["scaler"])

        print("Preloaded")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

    autocast_device_check = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype_check = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    accumulate_gradients = config['gradient_accumulation']
    accumulate_gradients_steps = config['accumulation_steps']
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(
            train_dataloader, desc=f"Processing Epoch: {epoch:02d}")

        for idx, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device, non_blocking=True) 
            decoder_input = batch['decoder_input'].to(device, non_blocking=True)  

            encoder_mask = torch.as_tensor(batch['encoder_mask']).to(device, non_blocking=True) 
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)  

            with torch.autocast(device_type="cuda",  # autocast_device_check,
                                dtype=autocast_dtype_check,
                                enabled=True):
                # run the tensors through the encoder, decoder and projection layer
                encoder_output = model.encode(src=encoder_input, src_mask=encoder_mask) 

                decoder_output = model.decode(encoder_output=encoder_output,
                                              src_mask=encoder_mask,
                                              tgt=decoder_input,
                                              tgt_mask=decoder_mask)

                proj_output = model.project(decoder_output)

                # compare output and label
                label = batch['label'].type(torch.LongTensor).to(device)  # (b, seq_len)

                # compute the loss using cross entropy
                loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
                if accumulate_gradients:
                    loss = loss / accumulate_gradients_steps

            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad(set_to_none=True)

            # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()
            if accumulate_gradients:
                if idx % accumulate_gradients_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            # if scheduler is not None:
            scheduler.step()
            lr_v = scheduler.get_last_lr()
            lrs.append(lr_v)
            if accumulate_gradients:
                batch_iterator.set_postfix(
                    {"Accumulated scaled loss": f"{loss.item() * accumulate_gradients_steps:8.5f}", "lr": f"{lr_v}"})
            else:
                batch_iterator.set_postfix({"loss": f"{loss.item():8.5f}", "lr": f"{lr_v}"})
            # else:
            #     batch_iterator.set_postfix({"loss": f"{loss.item() * accumulate_gradients_steps:8.5f}"})

            # log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            global_step += 1

        # Save the model at end of each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch'               : epoch,
            'model_state_dict'    : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step'         : global_step,
            'scaler'              : scaler.state_dict()
        }, model_filename)

    if run_val:
        run_validation(model=model,
                       val_dataloader=val_dataloader,
                       tokenizer_src=tokenizer_src,
                       tokenizer_tgt=tokenizer_tgt,
                       max_len=config['seq_len'],
                       device=device, 
                    #    print_msg=None, #lambda msg: batch_iterator.write(msg),
                       global_step=global_step,
                       writer=writer,
                    #    num_examples=2
                       )


# def run_validation(model,
#                    validation_ds,
#                    tokenizer_src,
#                    tokenizer_tgt,
#                    max_len,
#                    device,
#                    print_msg,
#                    global_step,
#                    writer,
#                    num_examples: 2):
#     model.eval()
#     count = 0

#     source_texts = []
#     excepted = []
#     predicted = []

#     try:
#         # get console window width
#         with os.popen('stty size', 'r') as console:
#             _, console_width = console.read().split()
#             console_width = int(console_width)
#     except:
#         console_width = 80

#     with torch.no_grad():
#         for batch in validation_ds:
#             count += 1
#             encoder_input = batch['encoder_input'].to(device, non_blocking=True)
#             encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)

#             # check that the batch siz is 1
#             assert encoder_input.size(0) == 1, "Batch size for val loader must be 1"

#             model_out = greedy_decode(model=model,
#                                       source=encoder_input,
#                                       source_mask=encoder_mask,
#                                       tokenizer_src=tokenizer_src,
#                                       tokenizer_tgt=tokenizer_tgt,
#                                       max_len=max_len,
#                                       device=device)

#             source_text = batch['src_text'][0]
#             target_text = batch['tgt_text'][0]
#             model_out_text = tokenizer_tgt.decode(
#                 model_out.detach().cpu().numpy())

#             source_texts.append(source_text)
#             excepted.append(target_text)
#             predicted.append(model_out_text)

#             if count <= num_examples:
#                 # print source, target and model output
#                 print('-' * console_width)
#                 print(f"===={f'SOURCE: ':>12}{source_text}")
#                 print(f"===={f'TARGET: ':>12}{target_text}")
#                 print(f"===={f'PREDICTED: ':>12}{model_out_text}")
#                 print('-' * console_width)

#     if writer:
#         # Character error rate
#         metric = torchmetrics.CharErrorRate()
#         cer = metric(predicted, excepted)
#         writer.add_scalar('validation CER', cer, global_step)
#         writer.flush()

#         # Word error rate
#         metric = torchmetrics.WordErrorRate()
#         wer = metric(predicted, excepted)
#         writer.add_scalar('validation WER', wer, global_step)
#         writer.flush()

#         # Compute BLEU metric
#         metric = torchmetrics.BLEUScore()
#         bleu = metric(predicted, excepted)
#         writer.add_scalar('validation BLEU', bleu, global_step)
#         writer.flush()
            
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    
    