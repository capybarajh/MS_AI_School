import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset) :
    def __init__(self, data_dir, transforms = None):
        self.label_dict = {"Normal": 0 , "Pneumonia_bacteria" : 1, "Pneumonia_virus" : 2}
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.jpeg"))
        self.transforms = transforms

    def __getitem__(self, item):
        path = self.data_dir[item]
        image = Image.open(path)
        image = image.convert("RGB")

        # label
        # ['./pneumonia_data/train', 'Normal', 'IM-0115-0001.jpeg']
        dir_name = path.split("\\")[1]
        label = int(self.label_dict[dir_name])

        if self.transforms is not None :
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.data_dir)



