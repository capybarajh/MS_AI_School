import glob
import os

from torch.utils.data import Dataset
from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

class My_ex01_customdata(Dataset) :
    def __init__(self, data_dir, transforms=None):
        # data_dir => ./dataset/train/
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.jpg"))
        self.transforms = transforms
        self.label_dict = {'Carpetweeds':0, 'Crabgrass':1,'Eclipta' : 2, 'Goosegrass':3 ,
                           'Morningglory':4, 'Nutsedge':5, 'PalmerAmaranth':6, 'Prickly Sida':7,
                           'Purslane':8, 'Ragweed':9, 'Sicklepod':10, 'SpottedSpurge':11,
                           'SpurredAnoda':12, 'Swinecress':13, 'Waterhemp':14
        }

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
