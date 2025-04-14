import glob
import os

from torch.utils.data import Dataset
from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

class My_ex02_customdata(Dataset) :
    def __init__(self, data_dir, transforms=None):
        # data_dir => ./org_dataset/Train
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.jpg"))
        self.transforms = transforms
        self.label_dict = {"Chickenpox":0, "Cowpox": 1,"Healthy":2, "HFMD": 3 , "Measles":4,
                           "Monkeypox":5}

    def __getitem__(self, item):
        image_path = self.data_dir[item]
        # ['./dataset/train', 'Carpetweeds', 'Carpetweeds_314.jpg']
        label_name = image_path.split("\\")[1]
        label = self.label_dict[label_name]

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transforms is not None :
            image= self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.data_dir)
