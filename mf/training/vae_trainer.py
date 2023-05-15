from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from einops import rearrange
import albumentations
from contextlib import nullcontext #Python 3.7 and above

class VAETraining:
    def __init__(self, model_holder, dataset, training_params):
        self.models = model_holder
        self.dataset = dataset
        self.training_params = training_params
        self.batch_size = training_params["batch_size"]
        self.accumulation_steps = training_params["gradient_accumulation_steps"]
        self.num_dataset_workers = training_params["dataset_workers"]
        self.repeats = training_params["repeats"] if "repeats" in training_params else 0

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

        # This literally won't work without xformers for some reason
        vae.set_use_memory_efficient_attention_xformers(True)

        optimizer = vae_wrapper.create_optimizer()
        steps_pbar = tqdm(range(len(dataloader) * (self.repeats + 1) * self.batch_size), position=0, leave=False, dynamic_ncols=True)
        for _ in range(self.repeats + 1):
            accum_loss = None
            accumed_steps = 0
            for step, batch in enumerate(dataloader):
                image_tensor = batch["image"].to(memory_format=torch.contiguous_format,dtype=vae_wrapper.get_datatype()).to(vae_wrapper.get_device())
                # Forward pass
                model_pred = self._encode_and_decode(image_tensor, vae_wrapper.do_train_encoder(), vae_wrapper.do_train_decoder())

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), image_tensor.float(), reduction="mean")
                if accum_loss is None:
                    accum_loss = loss
                else:
                    accum_loss += loss
                accumed_steps += 1

                if (step + 1) % self.accumulation_steps == 0:
                    accum_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    steps_pbar.set_postfix({"loss": accum_loss.detach().item()})
                    steps_pbar.update(self.accumulation_steps * self.batch_size)
                    accum_loss = None
                    accumed_steps = 0

            # Can happen when steps is not divisible by accumulation steps
            if accum_loss is not None:
                accum_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
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
            

