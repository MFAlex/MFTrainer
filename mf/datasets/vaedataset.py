import os
from tqdm.auto import tqdm
import PIL.Image as Image
import albumentations
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random

class MFVAEDataset(Dataset):
    def __init__(self, global_path, config):
        if "subfolder" in config and config["subfolder"] != "":
            self.path = os.path.join(global_path, config["subfolder"])
        else:
            self.path = global_path
        allowed_extensions = [".png", ".jpg", "webp"]
        self.files = [file for file in os.listdir(self.path) if file[-4:].lower() in allowed_extensions]
        self.shuffle_each_epoch = config["shuffle_each_epoch"] if "shuffle_each_epoch" in config else False
        self.resize = config["resize"]
        self.train_size = config["train_size"]
        assert self.resize < 1 or self.resize >= self.train_size, "Cannot resize an image to smaller than the training size"
        self.transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.0], [0.5])
            ]
        )
        self.cropper = albumentations.RandomCrop(height=self.train_size, width=self.train_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_path = os.path.join(self.path, self.files[i])
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self._resize(image)

        to_train = self.cropper(image=image)["image"]

        f32_img = (to_train/127.5 - 1.0).astype(np.float32)
        image_tensor = self.transformer(f32_img)
        data = dict()
        data["image"] = image_tensor
        return data
    
    def next_epoch(self):
        if self.shuffle_each_epoch:
            random.shuffle(self.files)

    def _resize(self, img):
        smallest_length = min(img.shape[0], img.shape[1])
        if self.resize > 1:
            # Resize to this many pixels
            smallest_length = int(max(smallest_length / self.resize, self.train_size))
            img = albumentations.SmallestMaxSize(max_size=smallest_length, interpolation=cv2.INTER_AREA)(image=img)["image"]
        elif self.resize > 0:
            # If from 0-1, treat as a ratio, but make sure it's not too small to train on
            smallest_length = int(max(smallest_length / self.resize, self.train_size))
            img = albumentations.SmallestMaxSize(max_size=smallest_length, interpolation=cv2.INTER_AREA)(image=img)["image"]
        return img

class SmartMFVAEDataset(MFVAEDataset):
    def __init__(self, global_path, config):
        super().__init__(global_path, config)
        self.real_cropper = self.cropper
        self.cropper = self._smart_crop_func
    
    def _smart_crop_func(self, image):
        """
        Takes the stddev of the inputs of each channel (r, g, b) of the image. Effectively the variance in the histogram
        Then if there is sufficient variation, return that sample. Otherwise try 50 random parts of the image until we find a good one,
        or return the best one
        """
        best = None
        best_stddev = 0
        for _ in range(50):
            sample = self.real_cropper(image=image)["image"]
            rgb_layers = np.moveaxis(sample, 2, 0)
            stddev1 = np.std(rgb_layers[0])
            stddev2 = np.std(rgb_layers[1])
            stddev3 = np.std(rgb_layers[2])
            stddev = max(stddev1, stddev2, stddev3)
            if stddev > 15:
                best = sample
                break
            elif stddev > best_stddev:
                best_stddev = stddev
                best = sample
        r = dict()
        r["image"] = best
        return r

class CenterMFVAEDataset(MFVAEDataset):
    def __init__(self, global_path, config):
        config["shuffle_each_epoch"] = False
        super().__init__(global_path, config)
        self.cropper = albumentations.CenterCrop(height=self.train_size, width=self.train_size)