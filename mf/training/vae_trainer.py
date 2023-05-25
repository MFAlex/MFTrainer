from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from contextlib import nullcontext #Python 3.7 and above
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning import Trainer, LightningModule
import math
import copy

class VAETraining:
    def __init__(self, model_holder, dataset, training_params):
        self.models = model_holder
        self.dataset = dataset
        self.training_params = training_params
        self.batch_size = training_params["batch_size"]
        self.accumulation_steps = training_params["gradient_accumulation_steps"]
        self.num_dataset_workers = training_params["dataset_workers"]
        self.repeats = training_params["repeats"] if "repeats" in training_params else 0
        self.train_enc = training_params["train_encoder"]
        self.train_dec = training_params["train_decoder"]

    def train(self):
        self.dataset.next_epoch()
        # Prepare data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_dataset_workers
        )

        vae_wrapper = self.models.get_vae()
        vae = vae_wrapper.get_model()

        vae.encoder.train() if self.train_enc else vae.encoder.eval()
        vae.decoder.train() if self.train_dec else vae.decoder.eval()

        # This literally won't work without xformers for some reason
        vae.set_use_memory_efficient_attention_xformers(True)

        optimizer = vae_wrapper.get_optimizer(train_enc=self.train_enc, train_dec=self.train_dec)
        steps_pbar = tqdm(range(len(self.dataset) * (self.repeats + 1)), position=0, leave=False, dynamic_ncols=True)
        for _ in range(self.repeats + 1):
            accum_loss = None
            accumed_steps = 0
            for step, batch in enumerate(dataloader):
                image_tensor = batch["image"].to(memory_format=torch.contiguous_format,dtype=vae_wrapper.get_datatype()).to(vae_wrapper.get_device())
                # Forward pass
                model_pred = self._encode_and_decode(image_tensor, self.train_enc, self.train_dec)

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), image_tensor.float(), reduction="mean")
                if accum_loss is None:
                    accum_loss = loss
                else:
                    accum_loss += loss
                accumed_steps += batch["image"].shape[0]

                if (step + 1) % self.accumulation_steps == 0:
                    accum_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps_pbar.set_postfix({"loss": accum_loss.detach().item()})
                    steps_pbar.update(accumed_steps)
                    accum_loss = None
                    accumed_steps = 0

            # Can happen when steps is not divisible by accumulation steps
            if accum_loss is not None:
                accum_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps_pbar.set_postfix({"loss": accum_loss.detach().item()})
                steps_pbar.update(accumed_steps)
                accum_loss = None
            self.dataset.next_epoch() # shuffle as a repeat can be considered an epoch

        # Done with the model. Bring it back to the CPU
        vae_wrapper.set_model_idle()
    
    def _encode_and_decode(self, image_tensor, train_encoder=True, train_decoder=True):
        vae = self.models.get_vae().get_model()
        with torch.no_grad() if not train_encoder else nullcontext():
            posterior = vae.encode(image_tensor).latent_dist
        
        latents = posterior.mode()
        # TODO normalize latents to simulate diffusion process without any offset noise training

        with torch.no_grad() if not train_decoder else nullcontext():
            decoded = vae.decode(latents).sample
        return decoded
            
class VAE_LearingRate:
    def __init__(self, model_holder, dataset, training_params):
        self.models = model_holder
        self.dataset = dataset
        self.training_params = training_params
        self.batch_size = training_params["batch_size"]
        self.accumulation_steps = training_params["gradient_accumulation_steps"]
        self.start_lr = training_params["start_lr"]
        self.end_lr = training_params["end_lr"]
        assert self.start_lr < self.end_lr
        self.num_steps = training_params["steps"]
        self.scale = training_params["scale"]
        assert self.scale in ["exponential", "linear"]

    def find(self):
        self.dataset.next_epoch()
        # Prepare dataset
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1
        )

        vae_wrapper = self.models.get_vae()
        vae_original = vae_wrapper.get_model()
        vae_original.to(torch.device("cpu")) # Put it in RAM

        # Load the same dataset for each LR test
        batches = []
        for batch in dataloader:
            batches.append(batch["image"].to(memory_format=torch.contiguous_format,dtype=vae_wrapper.get_datatype()))
        
        results = []
        lrs = []
        for test in tqdm(range(self.num_steps)):
            # Copy the model
            vae_clone = copy.deepcopy(vae_original)
            vae_clone.to(vae_wrapper.get_device())

            # Decide on the LR to test
            test_ratio = test / self.num_steps
            lr_range = self.end_lr - self.start_lr
            if test == 0:
                test_lr = self.start_lr
            elif self.scale == "exponential":
                test_lr = self.start_lr + lr_range * ((100 ** test_ratio) - 1) / 99
            else:
                test_lr = self.start_lr + lr_range * test_ratio
            # Set the model LR
            optimizer = vae_wrapper.create_optimizer(vae_clone, lr_override=test_lr)
            total_loss = 0
            for image_tensor in batches:
                image_tensor = image_tensor.to(vae_wrapper.get_device())
                model_pred = self._encode_and_decode(vae_clone, image_tensor, vae_wrapper.do_train_encoder(), vae_wrapper.do_train_decoder())
                loss = F.mse_loss(model_pred.float(), image_tensor.float(), reduction="mean")
                image_tensor = image_tensor.to(torch.device("cpu"))
                total_loss += loss.detach().item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            lrs.append(test_lr)
            results.append(total_loss)
            del vae_clone
        print(results)
        print(lrs)
        vae_wrapper.set_model_idle()
        # TODO apply learning rate

    def _encode_and_decode(self, vae, image_tensor, train_encoder=True, train_decoder=True):
        with torch.no_grad() if not train_encoder else nullcontext():
            posterior = vae.encode(image_tensor).latent_dist
        
        latents = posterior.mode()

        with torch.no_grad() if not train_decoder else nullcontext():
            decoded = vae.decode(latents).sample
        return decoded

class VAE_Validation:
    def __init__(self, model_holder, dataset):
        self.models = model_holder
        self.dataset = dataset

    def validate(self):
        self.dataset.next_epoch()
        # Prepare dataset
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

        vae_wrapper = self.models.get_vae()
        vae = vae_wrapper.get_model()
        total_loss = 0
        for batch in tqdm(dataloader):
            image_tensor = batch["image"].to(memory_format=torch.contiguous_format,dtype=vae_wrapper.get_datatype()).to(vae_wrapper.get_device())
            with torch.no_grad():
                predicted = self._encode_and_decode(vae, image_tensor)
                loss = F.mse_loss(predicted.float(), image_tensor.float(), reduction="mean").detach().item()
                total_loss += loss
        print("Total validation loss:", total_loss)
        vae_wrapper.set_model_idle()

    def _encode_and_decode(self, model, image_tensor):
        with torch.no_grad():
            posterior = model.encode(image_tensor).latent_dist
            latents = posterior.mode()
            decoded = model.decode(latents).sample
            return decoded