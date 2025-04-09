from typing import Any
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsDataset(Dataset):
    def __init__(self, file_paths: list, labels, transform=None):
        # 이번 예시에서는 간추린 파일 몇개만 직접 입력해서 사용
        self.file_paths = file_paths  
        # self.file_lists = os.listdir(file_paths)
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]
        file_path = self.file_paths[index]

        image = cv2.imread(file_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label
    
    def __len__(self):
        return len(self.file_paths)
    

if __name__ == "__main__":
    albumentations_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    dataset = AlbumentationsDataset(
        ["sample_data_01\\train\\dew\\2208.jpg",
         "sample_data_01\\train\\fogsmog\\4075.jpg",
         "sample_data_01\\train\\frost\\3600.jpg"],
         [0, 1, 2],
        transform=albumentations_transform
    )

    for image, label in dataset:
        print(f"Data of dataset : {image}, {label}")