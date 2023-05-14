import mf.model_utils as utils
import logging
import os

class ModelHolder:
    def __init__(self, vae=None, text_encoder=None, tokenizer=None, unet=None, scheduler=None):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler

    def _set(self, model, value):
        if model == "vae":
            self.vae = value
        elif model == "text_encoder":
            self.text_encoder = value
        elif model == "tokenizer":
            self.tokenizer = value
        elif model == "unet":
            self.unet = value
        elif model == "scheduler":
            self.scheduler = value
    
    def get_vae(self):
        if self.vae is None:
            raise Exception("Training tried to use vae, but this was not specified in config")
        else:
            return self.vae
        
    def get_text_encoder(self):
        if self.text_encoder is None:
            raise Exception("Training tried to use text_encoder, but this was not specified in config")
        else:
            return self.text_encoder
        
    def get_tokenizer(self):
        if self.tokenizer is None:
            raise Exception("Training tried to use tokenizer, but this was not specified in config")
        else:
            return self.tokenizer
        
    def get_unet(self):
        if self.unet is None:
            raise Exception("Training tried to use unet, but this was not specified in config")
        else:
            return self.unet
        
    def get_scheduler(self):
        if self.scheduler is None:
            raise Exception("Training tried to use scheduler, but this was not specified in config")
        else:
            return self.scheduler
    
    def save_models(self, path, which_models, datatype="float32"):
        dtype = utils.parse_datatype(datatype)
        if "vae" in which_models:
            self.vae.get_model().to(dtype=dtype).save_pretrained(os.path.join(path, "vae"))


class ModelManager:
    def __init__(self, path, hardware, type, training_config):
        assert training_config["mode"] in ["train", "eval"]
        self.path = path
        self.type = type
        self.loaded_model = False
        self.model_is_idle = True
        self.model_type = type
        self.training_config = training_config
        self.device = hardware[training_config["location"]]
        self.dtype = utils.parse_datatype(training_config["data_type"])
        self.idle_device = hardware["cpu"]
    
    def get_model(self):
        if not self.loaded_model:
            logging.debug(f"Moving module {self.type} from disk to cpu")
            self.model = utils.load_model(self.path, components=[self.model_type])[0]
            self.loaded_model = True
            self.model_is_idle = True
            if self.training_config["mode"] == "train":
                self.model.train()
            elif self.training_config["mode"] == "eval":
                self.model.eval()
            self.init_hooks(self.model)
        if self.model_is_idle:
            logging.debug(f"Moving module {self.type} from cpu to {self.training_config['location']}")
            self.model.to(self.device, dtype=self.dtype)
            self.model_is_idle = False
        return self.model
    
    def set_model_idle(self):
        logging.debug(f"Moving module {self.type} from {self.training_config['location']} to cpu")
        self.model.to(self.idle_device)
        self.model_is_idle = True

    def get_device(self):
        return self.device
    
    def get_datatype(self):
        return self.dtype

    def init_hooks(self, model):
        """
        Give the implementing models a chance to configure themselves when the model is first loaded
        """
        pass

    def create_optimizer(self):
        raise NotImplementedError()

class VAEManager(ModelManager):
    def __init__(self, model_path, hardware, training_config):
        super().__init__(model_path, hardware, "vae", training_config)
        validate_optimizer(training_config)
    
    def init_hooks(self, model):
        if self.training_config["optimizations"]["slicing"]:
            model.enable_slicing()
        else:
            model.disable_slicing()
        if self.training_config["mode"] == "train":
            train_enc = self.training_config["train_encoder"]
            train_dec = self.training_config["train_decoder"]
            assert train_enc or train_dec
            if not train_enc:
                model.encoder.eval()
                model.encoder.train = lambda x: x
            if not train_dec:
                model.decoder.eval()
                model.decoder.train = lambda x: x
    
    def create_optimizer(self):
        train_enc = self.training_config["train_encoder"]
        train_dec = self.training_config["train_decoder"]
        if train_enc and train_dec:
            params = self.model.parameters()
        elif train_enc:
            params = self.model.encoder.parameters()
        elif train_dec:
            params = self.model.decoder.parameters()
        return make_optimizer(params, self.training_config)
    
    def do_train_encoder(self):
        return self.training_config["train_encoder"]
    
    def do_train_decoder(self):
        return self.training_config["train_decoder"]

def make_optimizer(params, config):
    optimizer_config = config["optimizer"]
    optimizer_type = optimizer_config["type"]
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"] if "weight_decay" in optimizer_config else 0.0
    if optimizer_type in ["8bit-adamw", "lion", "8bit-lion", "8bit-adam"]:
        import bitsandbytes as bnb
        if optimizer_type == "8bit-adamw":
            opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "lion":
            opt = bnb.optim.Lion(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "8bit-lion":
            opt = bnb.optim.Lion8bit(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "8bit-adam":
            opt = bnb.optim.Adam8bit(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "lion-pytorch":
        from lion_pytorch import Lion
        opt = Lion(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type in ["adamw", "adam"]:
        import torch.optim
        if optimizer_type == "adamw":
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        print(f"Unknown optimizer '{optimizer_type}'. Defaulting to 'adamw'")
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return opt

def validate_optimizer(config):
    t = config["optimizer"]["type"]
    assert t in ["adamw", "8bit-adamw", "lion", "8bit-lion", "adam", "8bit-adam", "lion-pytorch"]
    dtype = config["data_type"]
    if dtype == "bfloat16":
        assert t not in ["8bit-lion", "8bit-adamw", "8bit-adam", "lion"], f"'{dtype}' is not supported with optimizer '{t}'"