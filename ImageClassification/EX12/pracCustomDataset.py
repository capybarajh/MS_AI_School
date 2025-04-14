from torch.utils.data import Dataset
import glob
import os
import cv2
import pandas as pd


class SportsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        '''
        data_dir: 데이터셋의 경로
        transform: augmentation 등에 필요한 transform 정의
        '''
        self.file_list = glob.glob(os.path.join(data_dir, "*", "*.jpg"))
        # 파일 경로를 받아와서 하위 폴더 밑의 이미지 파일을 list로 받도록 저장
        self.transform = transform
        # 인자로 받은 transform을 내부 값으로 저장

        # label dictionary를 생성자 호출 시 만들어놓고 저장해놓을 필요가 있음
        self.label_dict = {}
        folder_name_list = glob.glob(os.path.join(data_dir, "*")) # data_dir 하위의 폴더 이름만을 가져오도록
        # 단, 위의 folder_name_list는 상위 폴더 이름까지 모두 붙은 상태
        for i, folder_name in enumerate(folder_name_list): # for 문을 순회하면서 순번도 같이 반환함
            folder_name = os.path.basename(folder_name) # 스포츠 이름만을 떼어놓도록
            # print(f"{i}번째 요소 : {folder_name}") # folder_name이 label_dict의 key값이 되도록
            # value는 이 folder_name의 순번이 되도록
            self.label_dict[folder_name] = i

    def construct_dataset_by_csv(self, csv_dir, transform=None):
        # csv 파일을 읽어서 구성하는 경우
        # 편의상 별도 함수로 나눴지만 이 함수가 생성자가 될 것임 !!!
        self.csv_data = pd.read_csv(csv_dir)
        self.file_list_by_csv = self.csv_data['filepaths'] 
        # csv 파일 내의 filepaths 칼럼에 파일명들이 저장되어 있으므로, 해당 내용을 데이터셋으로 사용
        # 나중에 __len__ 함수에서 len(self.file_list_by_csv)

        # label이 폴더 이름이 아닌, csv 내부에 직접 명시되어 있으므로 해당 내용을 사용하면 됨
        self.label_column = self.csv_data['labels']
        # class id가 이미 숫자로 표기되어 있으므로, dictionary 구성이 필요 없음
        self.class_id = self.csv_data['class id']

        # csv 파일 내에 .lnk 파일이 포함된 오류가 있으므로, 해당 파일 걸러내는 작업 진행

        #

        self.transform = transform


    def __getitem__(self, index):
        '''
        index: 데이터 로더 등이 요구하는 파일의 번호
        '''
        img = cv2.imread(self.file_list[index])  # 요구하는 파일을 cv2 이미지로 로딩
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2로 불러온 이미지는 BGR 채널이므로, RGB로 변환 필요

        label = os.path.dirname(self.file_list[index])  # 이렇게 불러올 경우 상위 폴더 이름까지 모두 딸려옴
        label = os.path.basename(label)  # 상위 폴더 이름을 모두 제거한 마지막 이름 (스포츠 이름명으로 된 폴더)

        # label이 숫자값으로 반환되어야 하기 때문에 dictionary를 이용해서 
        # 스포츠 이름 문자에 해당하는 숫자를 반환하도록 할 필요가 있음
        label = self.label_dict[label]

        # csv로 데이터를 구성할 경우, class id가 이미 명시되어 있으므로 그대로 반환값으로 사용
        # label = self.class_id[index]

        if self.transform is not None:
            img = self.transform(image=img)['image'] # albumentation transform 사용시
        # print(index)
        return img, label

    def __len__(self):
        return len(self.file_list)
    

if __name__ == "__main__":
    # 이번 파일을 테스트하기 위해 실행시 작동하는 코드
    dataset = SportsDataset("./US_license_plates_dataset/train")
    # dataset.construct_dataset_by_csv("./US_license_plates_dataset/sports.csv")
    for item in dataset:
        # __getitem__을 테스트해보는 구간
        _, label = item  # __getitem__의 return값인 item은 img, label로 구성되어 있으므로, 해체
        # print(label)

    # print(len(dataset))