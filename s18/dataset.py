import torch
import torch.nn as nn
from torch.utils.data import Dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
class BillingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int8)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int8)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int8)
        
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]

        # get source and target texts
        src_txt = src_target_pair['translation'][self.src_lang]
        tgt_txt = src_target_pair['translation'][self.tgt_lang]

        # transform texts into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_txt).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_txt).ids

        # add SOS, EOS and PAD to each sentence
        # calculate number of pads needed for encoder and decoder separately
        # eg: 350(max seq_len) - 5(input seq or sentence len) - 2(sos and eos) = 343
        # enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # consider SOS and EOS
        #
        # # consider SOS only as EOS is in the label
        # dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # make sure the padded sequence is not negative, seq_len should be calculated separately beforehand for max value
        # if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
        #     raise ValueError("Sentence too long")

        # Add <s> and </s> token and <pad>, with calculation for number of <pad>s needed
        # Encoder structure is: <SOS>, <input_token_ids>, <EOS>, <PAD>, ... <PAD>
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int32),
                self.eos_token,
                # torch.tensor([self.pad_token] *
                #              enc_num_padding_tokens, dtype=torch.int32)
            ], dim=0
        )

        # Decoder: add only <SOS>, <input_token_ids> and <PAD>...<PAD> , as <EOS> is to be predicted by the decoder
        # <SOS>, <input_token_ids>, <PAD>, ... <PAD>
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int32),
                # torch.tensor([self.pad_token] *
                #              dec_num_padding_tokens, dtype=torch.int32)
            ], dim=0
        )

        # for label, add only <EOS> token with input and pads
        # <decoder input token ids>, <EOS>, <PAD>, ... <PAD>
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int32),
                self.eos_token,
                # torch.tensor([self.pad_token] *
                #              dec_num_padding_tokens, dtype=torch.int32)

            ]
        )

        # checks for dimensions
        # assert encoder_input.size(0) == self.seq_len
        # assert decoder_input.size(0) == self.seq_len
        # assert label.size(0) == self.seq_len

        # enc_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        # dec_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

        return {
            "encoder_input": encoder_input,  # seq_len
            "decoder_input": decoder_input,  # seq_len
            # (1, 1, seq_len)
            # "encoder_mask" : enc_mask,
            # "decoder_mask" : dec_mask,
            # (1, seq_len) & (1, seq_len, seq_len)
            "label"        : label,  # (seq_len)
            "src_text"     : src_txt,
            "tgt_text"     : tgt_txt,
            "encoder_token_len": len(encoder_input),
            "decoder_token_len": len(decoder_input),
            'pad_token': self.pad_token
        }
    # def __getitem__(self, idx):
    #     src_tgt_pair = self.ds[idx]
    #     src_text = src_tgt_pair['translation'][self.src_lang]
    #     tgt_text = src_tgt_pair['translation'][self.tgt_lang]
        
    #     enc_input_tokens = self.tokenizer_src.encode(src_text).ids
    #     dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
    #     enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
    #     dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
    #     #For encoding, we PAD both SOS and EOS. For decoding, we only pad SOS.
    #     #THe model is required to predict EOS and stop on its own.
        
    #     #Make sure that padding is not negative (ie the sentance is too long)
    #     if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
    #         raise ValueError("Sentence too long")
            
    #     encoder_input = torch.cat(
    #         [
    #             self.sos_token,
    #             torch.tensor(enc_input_tokens, dtype = torch.int64),
    #             self.eos_token,
    #             # torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype = torch.int64)
    #         ],
    #         dim =  0,
    #     )
        
    #     decoder_input = torch.cat(
    #         [
    #             self.sos_token,
    #             torch.tensor(dec_input_tokens, dtype = torch.int64),
    #             # torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64)
    #         ],
    #         dim = 0,
    #     )
        
    #     label = torch.cat(
    #         [
    #             torch.tensor(dec_input_tokens, dtype = torch.int64),
    #             self.eos_token,
    #             # torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64),
    #         ],
    #         dim = 0,
    #     )
        
    #     # assert encoder_input.size(0) == self.seq_len
    #     # assert decoder_input.size(0) == self.seq_len
    #     # assert label.size(0) == self.seq_len
        
    #     return {
    #         "encoder_input": encoder_input,
    #         "decoder_input": decoder_input,
    #         "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), 
    #         # encoder mask: (1, 1, seq_len) -> Has 1 when there is text and 0 when there is pad (no text)
            
    #         "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
    #         # (1, seq_len) and (1, seq_len, seq_len)
    #         # Will get 0 for all pads. And 0 for earlier text.
    #         "label": label,
    #         "src_text": src_text,
    #         "tgt_text": tgt_text
    #         }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal = 1).type(torch.int)
    #This will get the upper traingle values
    return mask == 0
    
    
    
