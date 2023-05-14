import diffusers, transformers
import torch
import json
import mf.model_manager as models_manager
import mf.model_utils as utils
import os
import argparse

from mf.training.vae_trainer import VAETraining

def train(config_file_path, model_path, out_model_path, dataset_path):
    # Initialise some important state variables
    model_holder = models_manager.ModelHolder()
    hardware = dict()

    print(f"Parsing {config_file_path}")
    f = open(config_file_path, "r")
    config = json.load(f)
    f.close()

    if out_model_path == None:
        out_model_path = model_path

    # Load hardware definitions
    hardware["cpu"] = torch.device("cpu") #always have CPU available by default
    for key in config["hardware"].keys():
        if config["hardware"][key]["type"] == "cuda":
            hardware[key] = torch.device(f"cuda:{config['hardware'][key]['gpuid']}")

    # Load models
    needed_models = list(config["models"].keys())
    for model in needed_models:
        if model == 'vae':
            model_holder._set("vae", models_manager.VAEManager(model_path, hardware, config["models"][model]))

    datasets = dict()
    for key in config["datasets"].keys():
        datasets[key] = utils.construct_dataset_from_config(config["datasets"][key], dataset_path)

    print("Preparations complete. Let's do this!")
    for i, stage in enumerate(config["tasks"]):
        try:
            print(f"** EXECUTING TASK #{i} **")
            if stage["type"] == "vae_training":
                VAETraining(model_holder, datasets[stage["dataset"]]).train()
            elif stage["type"] == "save":
                to_save = stage["models"]
                dtype = stage["data_type"]
                model_holder.save_models(out_model_path, to_save, dtype)
        except KeyboardInterrupt:
            print(f"Interrupt command received. Interrupting current stage of type '{stage['type']}'")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="MFTrainer command line options")
    argparser.add_argument("--model", type=str, default=None, required=True, help="The path to the root directory of the huggingface-format model")
    argparser.add_argument("--output", type=str, default=None, required=False, help="The path to the root directory of the huggingface-format model to write to. Defaults to ")
    argparser.add_argument("--config", type=str, default=None, required=True, help="The path to the configuration file with what to train")
    argparser.add_argument("--dataset", type=str, default=None, required=True, help="The path for the root directory of the dataset to train on")

    args = argparser.parse_args()
    train(args.config, args.model, args.output, args.dataset)