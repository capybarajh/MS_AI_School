import os
import cv2
import glob
from torch.utils.data import Dataset

class MyUSLicensePlatesDataset(Dataset) :

    def __init__(self, data_dir, transform=None):
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.jpg"))
        self.transform = transform
        self.label_dict = self.create_label_dict()

    def create_label_dict(self):
        label_dict = {}
        for filepath in self.data_dir :
            label = os.path.basename(os.path.dirname(filepath))
            if label not in label_dict :
                label_dict[label] = len(label_dict)

        return label_dict

    def __getitem__(self, item):
        image_filepath = self.data_dir[item]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = os.path.basename(os.path.dirname(image_filepath))
        label_idx = self.label_dict[label]

        if self.transform is not None :
            image = self.transform(image=image)['image']

        return image, label_idx

    def __len__(self):

        return len(self.data_dir)