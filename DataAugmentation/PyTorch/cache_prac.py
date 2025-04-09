import torch
import os
import glob
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from custom_dataset_prac import CustomImageDataset

def is_grayscale(img):
    return img.mode == 'L'

class CachedCustomImageDataset(Dataset):
    def __init__(self, image_paths, transform = None):
        self.image_paths = glob.glob(os.path.join(image_paths, "*", "*", "*.jpg"))
        self.transform = transform
        self.label_dict = {"dew": 0, "fogsmog": 1, "frost": 2, "glaze": 3, "hail": 4,
                           "lightning": 5, "rain": 6, "rainbow": 7, "rime": 8, "sandstorm": 9,
                           "snow": 10}
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            image, label = self.cache[index]
        else:
            image_path: str = self.image_paths[index]
            image = Image.open(image_path).convert("RGB")

            if not is_grayscale(image):
                folder_name = image_path.split("\\")
                folder_name = folder_name[-2]

                label = self.label_dict[folder_name]

                self.cache[index] = (image, label)

            else:
                print(f"{image_path} 파일은 흑백 이미지입니다.")
                return None, None
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)
    

if __name__ == "__main__":
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image_paths = "./sample_data_01/"

    cached_dataset = CachedCustomImageDataset(image_paths, tf)
    cached_dataloader = DataLoader(cached_dataset, batch_size = 64, shuffle=True)

    not_cached_dataset = CustomImageDataset(image_paths, tf)
    not_cached_dataloader = DataLoader(not_cached_dataset, batch_size = 64, shuffle = True)

    c_start_time = time.time()
    for images, labels in cached_dataloader:
        pass
    print(f"캐시된 클래스 : {time.time() - c_start_time} 초 소모")

    nc_start_time = time.time()
    for images, labels in not_cached_dataloader:
        pass
    print(f"캐시되지 않은 클래스 : {time.time() - nc_start_time} 초 소모")

    c_reuse_start_time = time.time()
    for images, labels in cached_dataloader:
        pass
    print(f"캐시된 클래스 재사용 : {time.time() - c_reuse_start_time} 초 소모")

    nc_reuse_start_time = time.time()
    for images, labels in not_cached_dataloader:
        pass
    print(f"캐시되지 않은 클래스 재사용 : {time.time() - nc_reuse_start_time} 초 소모")