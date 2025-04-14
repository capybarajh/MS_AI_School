from torch.utils.data import Dataset
import glob
import os
import cv2
import pandas as pd


class MealDataset(Dataset):
    def __init__(self, root_path, mode, transform=None):
        '''
        root_path: train_set, val_set 폴더를 포함하고 있는 최상위 폴더
        mode: train 또는 val
        '''

        # self.data_files = glob.glob()
        csv_file_path = os.path.join(root_path,
                                     f"{mode}_labels.csv")
        self.root_path = root_path
        # 최상위 폴더만 지정한 후, 
        # 하위 파일 및 폴더는 mode 이름으로 구분되어 있으므로
        # mode 인자를 통해 자동으로 갖고 올 수 있도록 사용

        self.csv_data = pd.read_csv(csv_file_path)
        self.mode = mode

        self.file_list = self.csv_data['img_name'].to_list()
        self.label_idx = self.csv_data['label'].to_list()
        # csv 파일을 읽은 pandas column을 list로 변환해서 저장
        self.transform = transform

        print(self.csv_data)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        # file_name은 단순한 파일 이름이기 때문에
        file_path = os.path.join(self.root_path, 
                                 f"{self.mode}_set",
                                 f"{self.mode}_set",
                                 file_name)
        # (mode)_set 폴더를 os.path.join 함수로 붙임
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # albumentation 라이브러리에 사용하기 위해 RGB로 변경

        if self.transform is not None:
            img = self.transform(image=img)['image']
            # albumentation transform의 경우 상기와 같은 형식으로 이미지를 넣어줘야
            # 변환이 적용된 이미지를 반환받을 수 있음

        label = self.label_idx[index]
        # csv 파일에 이미 숫자값으로 존재하기 때문에 해당 값 그대로 사용

        return img, label

    def __len__(self):
        return len(self.file_list)


if __name__ == "__main__":
    dataset = MealDataset("./meal_set/", mode="val")
    for item in dataset:
        _, label = item
        print(label)