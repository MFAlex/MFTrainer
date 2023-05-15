import os, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import importlib
import torch
import open_clip.model as openclipmodel

def load_model(path, components = ["vae", "text_encoder", "tokenizer", "unet", "scheduler"]):
    assert os.path.exists(path)
    outputs = []
    for component in components:
        model = None
        if component == "vae":
            model = AutoencoderKL.from_pretrained(path, subfolder="vae")
        elif component == "text_encoder":
            model = CLIPTextModel.from_pretrained(path, subfolder="text_encoder")
        elif component == "tokenizer":
            model = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer", use_fast=False)
        elif component == "unet":
            model = UNet2DConditionModel.from_pretrained(path, subfolder="unet")
        elif component == "scheduler":
            model = DDPMScheduler.from_pretrained(path, subfolder="scheduler")
        elif component == "openclip":
            #model = factory.load_openai_model(os.path.join(path, "text_encoder", "pytorch_model.bin"))
            state_dict = torch.load(os.path.join(path, "text_encoder", "pytorch_model.bin"), map_location="cpu")
            model = openclipmodel.build_model_from_openai_state_dict(state_dict)
            print(model)
            
        assert model != None, "model \"" + component + "\" unrecognised or not present at the given path"
        outputs.append(model)
    return outputs

def construct_dataset_from_config(entry, globalpath):
    assert "class" in entry
    print(f"Instantiating dataset {entry['class']}")
    return get_obj_from_str(entry["class"])(globalpath, entry)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def parse_datatype(dtype):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    assert dtype in dtype_map.keys(), f"Unrecognised datatype {dtype}"
    return dtype_map[dtype]