import os.path

import torch
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset

# input in batch index -> model
def collate_fn(batch) :

    images, targets_boxes, targets_labels = tuple(zip(*batch))

    # image list image -> torch.stack use one tensor dim = 0
    images = torch.stack(images, 0)
    targets = [] # [] >> targets_boxes, targets_labels

    # targets_boxes
    for i in range(len(targets_boxes)) :

        target = {
            "boxes" : targets_boxes[i],
            "labels" : targets_labels[i]
        }
        targets.append(target)

    return images, targets

class CustomDataset(Dataset) :
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.image_path = sorted(glob.glob(os.path.join(root, "*.png")))

        if train :
            self.boxes = sorted(glob.glob(os.path.join(root, "*.txt")))

    def parse_boxes(self, box_path):
        with open(box_path, 'r', encoding='utf-8') as f :
             lines = f.readlines()

        boxes = []
        labels = []

        for line in lines :
            values = list(map(float, line.strip().split(' ')))
            # [25.0, 345.0, 405.0, 629.0, 405.0, 629.0, 748.0, 345.0, 748.0]
            class_id = int(values[0])
            x_min, y_min = int(round(values[1])), int(round(values[2]))
            x_max = int(round(max(values[3], values[5], values[7])))
            y_max = int(round(max(values[4], values[6], values[8])))

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        return torch.tensor(boxes, dtype=torch.float32), \
            torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, item):
        img_path = self.image_path[item]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]

        if self.train :
            box_path = self.boxes[item]
            boxes, labels = self.parse_boxes(box_path)
            labels += 1 # Background = 0

            if self.transforms is not None :
                transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
                img, boxes = transformed['image'], transformed['bboxes']
                labels = transformed['labels']

            return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

        else :
            if self.transforms is not None :
                transformed = self.transforms(image=img)
                img = transformed['image']

            file_name = img_path.split('/')[-1]
            return file_name, img, width, height

    def __len__(self):
        return len(self.image_path)
