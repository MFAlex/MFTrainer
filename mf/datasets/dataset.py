import os
from tqdm.auto import tqdm
import PIL.Image as Image
import albumentations
import cv2
import numpy as np

class MFDataset:
    def __init__(self, global_path, config):
        assert "size" in config
        self.path = os.path.join(global_path, config["subfolder"])
        allowed_extensions = [".png", ".jpg", "webp"]
        files = [file for file in os.listdir(self.path) if file[-4:].lower() in allowed_extensions]
        img_size = config["size"]
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA)
        self.crop_algo = albumentations.CenterCrop(height=img_size, width=img_size)

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


    def get_num_items(self):
        return len(self.data)
    
    def get_image(self, i):
        return self.data[i]
    
    def get_prompt(self, i):
        raise NotImplementedError("This dataset does not use prompts")