import os
from tqdm.auto import tqdm
import PIL.Image as Image
import albumentations
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from mf.model_utils import construct_dataset_from_config
import random

class MFDataset(Dataset):
    def __init__(self, global_path, config):
        assert "size" in config
        if "subfolder" in config and config["subfolder"] != "":
            self.path = os.path.join(global_path, config["subfolder"])
        else:
            self.path = global_path
        allowed_extensions = [".png", ".jpg", "webp"]
        files = [file for file in os.listdir(self.path) if file[-4:].lower() in allowed_extensions]
        img_size = config["size"]
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA)
        self.crop_algo = albumentations.CenterCrop(height=img_size, width=img_size)
        self.transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.0], [0.5])
            ]
        )

        self.data = []
        print(f"Preloading dataset of {len(files)} images in '{self.path}'")
        for file in tqdm(files):
            image = Image.open(os.path.join(self.path, file))
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
            image = self.image_rescaler(image=image)["image"]
            image = self.crop_algo(image=image)["image"]
            self.data.append(image)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.data[i]
        f32_img = (img/127.5 - 1.0).astype(np.float32)
        image_tensor = self.transformer(f32_img)

        data = dict()
        data["image"] = image_tensor
        return data
    
    def next_epoch(self):
        pass

class ConcatDataset(Dataset):
    def __init__(self, global_path, config):
        datasets = config["datasets"]
        self.shuffle_each_epoch = config["shuffle_each_epoch"] if "shuffle_each_epoch" in config else False
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        self.datasets = [None] * len(datasets)
        self.data = []
        for i, dataset in enumerate(datasets):
            d = construct_dataset_from_config(dataset, global_path)
            self.datasets[i] = d
            for j in range(len(d)):
                self.data.append((i, j))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        entry = self.data[i]
        return self.datasets[entry[0]][entry[1]]
    
    def next_epoch(self):
        if self.shuffle_each_epoch:
            random.shuffle(self.data)