from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torchvision import transforms
import albumentations
from contextlib import nullcontext #Python 3.7 and above

class VAETraining:
    def __init__(self, model_holder, dataset):
        self.models = model_holder
        self.dataset = dataset

    def train(self):
        vae_wrapper = self.models.get_vae()
        vae = vae_wrapper.get_model()

        # This literally won't work without xformers for some reason
        vae.set_use_memory_efficient_attention_xformers(True)

        optimizer = vae_wrapper.create_optimizer()
        steps_pbar = tqdm(range(self.dataset.get_num_items()), position=1, leave=False, dynamic_ncols=True)
        for step in range(self.dataset.get_num_items()):

            # Get image in correct format and on correct device
            rgb_image = self.dataset.get_image(step)
            image_tensor = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )((rgb_image/127.5 - 1.0).astype(np.float32))
            image_tensor = image_tensor[None, ...]
            image_tensor = image_tensor.to(memory_format=torch.contiguous_format,dtype=vae_wrapper.get_datatype()).to(vae_wrapper.get_device())
            # Forward pass
            model_pred = self._encode_and_decode(image_tensor, vae_wrapper.do_train_encoder(), vae_wrapper.do_train_decoder())

            # Calculate loss
            loss = F.mse_loss(model_pred.float(), image_tensor.float(), reduction="mean")
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            steps_pbar.set_postfix({"loss": loss.detach().item()})
            steps_pbar.update(1)
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
            

