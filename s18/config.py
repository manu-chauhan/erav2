from pathlib import Path

def get_config():
    return {
        "N":6,
        "h":8,
        "batch_size": 64,
        "pct_start":0.3,
        'anneal_strategy'   : "linear",
        "num_epochs": 20,
         'three_phase'       : True,
        # "lr": 10**-4,
        "max_lr": 10**-3,
        "initial_div_factor":100,
        "final_div_factor":100,
        'param_sharing'     : True,
        'gradient_accumulation': False,
        'accumulation_steps': 4,
        "seq_len": 350,
        "d_model": 512,
        "d_ff": 2048,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": True,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
        }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)