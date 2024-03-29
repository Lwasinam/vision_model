from pathlib import Path

def get_config():
    return {
        "batch_size":2,
        "num_epochs": 100,
        "lr": 50**-4,
        "seq_len": 150,
        "d_model": 768,
        "lang_src": "0",
        "lang_tgt": "1",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel",
        'project_name': 'proj1'
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


