import os

from torch.utils.data import Dataset
from PIL import Image
import glob

class CustomDataset_ex02(Dataset) :
    def __init__(self, data_dir, transform=None):
        # data_dir = ./data_art/train/
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.png"))
        self.transform = transform
        self.label_dict = {"Abstract" : 0 , "Cubist" : 1, "Expressionist" : 2,
                           "Impressionist" : 3, "Landscape" : 4, "Pop Art":5,
                           "Portrait" : 6, "Realist" :7, "Still Life" : 8,
                           "Surrealist" : 9}

    def __getitem__(self, item):
        image_path = self.data_dir[item]
        ##### 이미지 파일을 로드할 때 발생할 수 있는 에러를 처리
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        ###############################################
        image = Image.open(image_path)
        image = image.convert("RGB")
        label_name = image_path.split("\\")[1]
        label = self.label_dict[label_name]

        if self.transform is not None :
            image = self.transform(image)

        return image ,label, image_path

    def __len__(self):
        return len(self.data_dir)

# test = CustomDataset("./data_art/train" , transform=None)
#
# for i in test :
#     pass