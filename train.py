import diffusers, transformers
import torch
import json
import mf.model_manager as models_manager
import mf.model_utils as utils
import os
import argparse

from mf.training.vae_trainer import VAETraining, VAE_LearingRate, VAE_Validation

def do_task(task, model_holder, datasets, out_model_path):
    try:
        print(f"** EXECUTING TASK {task['type']} **")
        if task["type"] == "vae_training":
            VAETraining(model_holder, datasets[task["dataset"]], task["training_params"]).train()
        elif task["type"] == "vae_find_learning_rate":
            VAE_LearingRate(model_holder, datasets[task["dataset"]], task["training_params"]).find()
        elif task["type"] == "vae_validation":
            VAE_Validation(model_holder, datasets[task["dataset"]]).validate()
        elif task["type"] == "save":
            to_save = task["models"]
            dtype = task["data_type"] if "data_type" in task else "float32"
            with_ckpt = task["also_save_ckpt"] if "also_save_ckpt" in task else False
            version_str = None
            if "save_versions" in task and task["save_versions"]:
                import datetime
                version_str = datetime.datetime.now().strftime("%d_%b__%H_%M")
            model_holder.save_models(out_model_path, to_save, dtype, version=version_str, ckpt=with_ckpt)
        elif task["type"] == "loop":
            subtasks = task["tasks"]
            count = task["num_loops"]
            for _ in range(count):
                for t in subtasks:
                    do_task(t, model_holder, datasets, out_model_path)
    except KeyboardInterrupt:
        print(f"Interrupt command received. Interrupting current stage of type '{task['type']}'")

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
        elif model == 'openclip':
            model = utils.load_model(model_path, ["openclip"])
            print(model)

    datasets = dict()
    for key in config["datasets"].keys():
        datasets[key] = utils.construct_dataset_from_config(config["datasets"][key], dataset_path)

    print("Preparations complete. Let's do this!")
    for stage in config["tasks"]:
        do_task(stage, model_holder, datasets, out_model_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="MFTrainer command line options")
    argparser.add_argument("--model", type=str, default=None, required=True, help="The path to the root directory of the huggingface-format model")
    argparser.add_argument("--output", type=str, default=None, required=False, help="The path to the root directory of the huggingface-format model to write to. Defaults to ")
    argparser.add_argument("--config", type=str, default=None, required=True, help="The path to the configuration file with what to train")
    argparser.add_argument("--dataset", type=str, default=None, required=True, help="The path for the root directory of the dataset to train on")

    args = argparser.parse_args()
    train(args.config, args.model, args.output, args.dataset)