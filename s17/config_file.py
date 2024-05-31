from pathlib import Path

def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": False,
        "param_sharing": True,
         'anneal_strategy'   : "linear",
           'three_phase'       : True,
            'gradient_accumulation': False,
            'accumulation_steps': 4,
             'h': 8,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"}

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)