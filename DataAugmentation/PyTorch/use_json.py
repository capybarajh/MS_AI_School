import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os

class JsonCustomDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.transform = transform
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        img_path = self.data[index]['filename']
        img_path = os.path.join("이미지 폴더", img_path)

        # 이미지 읽기
        # image = Image.open(img_path).convert('RGB')

        # 바운딩 박스 정보 읽기
        bboxes = self.data[index]['ann']['bboxes']
        labels = self.data[index]['ann']['labels']

        # 바운딩 박스 정보를 Tensor로 변환
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        # 이미지 변환
        # if self.transform is not None:
            # image = self.transform(image)
    
        return img_path, {'boxes': bboxes, 'labels': labels}
    
    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    dataset = JsonCustomDataset("./test.json", transform=None)

    for item in dataset:
        print(f"Data of dataset : {item}")