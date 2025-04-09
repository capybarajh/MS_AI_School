from typing import Any
import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform = None):
        self.image_paths = glob.glob(os.path.join(image_paths, "*", "*", "*.jpg"))
        self.transform = transform
        self.label_dict = {"dew": 0, "fogsmog": 1, "frost": 2, "glaze": 3, "hail": 4,
                           "lightning": 5, "rain": 6, "rainbow": 7, "rime": 8, "sandstorm": 9,
                           "snow": 10}

    def __getitem__(self, index):
        image_path: str = self.image_paths[index]
        image = Image.open(image_path)
        folder_name = image_path.split("\\")
        folder_name = folder_name[2]

        lable = self.label_dict[folder_name]

        if self.transform:
            image = self.transform(image)

        return image, lable
    
    def __len__(self):
        return len(self.image_paths)
    
if __name__ == "__main__":
    image_paths = "./sample_data_01/"
    dataset = CustomImageDataset(image_paths)

    for item in dataset:
        print(f"Data and label : {item}")