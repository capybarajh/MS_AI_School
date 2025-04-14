from torch.utils.data import Dataset
import glob
import os
import cv2
import pandas as pd


class SportsDataset(Dataset):
    def __init__(self, csv_dir, mode="train", transform=None):
        '''
        csv_dir: csv 파일의 경로
        mode: 데이터셋의 모드. 'train', 'val', 'test' 세 가지가 가능
        transform: augmentation이나 Tensor화 등에 필요한 transform
        '''
        # csv 파일을 읽어서 구성하는 경우
        self.csv_data = pd.read_csv(csv_dir)

        self.csv_data = self.csv_data.loc[self.csv_data['data set'] == mode]

        # csv 파일 내에 .lnk 파일이 포함된 오류가 있으므로, 해당 파일 걸러내는 작업 진행
        for i, item in enumerate(self.csv_data['filepaths']):
            if item.endswith(".lnk"):
                # print(f"{i}번째 index에서 .lnk 발견됨")
                self.csv_data.drop(index=i, axis=0, inplace=True)
                self.csv_data.reset_index(inplace=True)
                # .lnk 파일이 들어있는 행을 삭제
        
        self.file_list_by_csv = self.csv_data['filepaths'].to_list()
        # csv 파일 내의 filepaths 칼럼에 파일명들이 저장되어 있으므로, 해당 내용을 데이터셋으로 사용
        # 나중에 __len__ 함수에서 len(self.file_list_by_csv)

        # label이 폴더 이름이 아닌, csv 내부에 직접 명시되어 있으므로 해당 내용을 사용하면 됨
        self.label_column = self.csv_data['labels'].to_list()
        # class id가 이미 숫자로 표기되어 있으므로, dictionary 구성이 필요 없음
        self.class_id = self.csv_data['class id'].to_list()

        # .to_list()로 리스트로 바꾸는 이유:
        # Dataset이 정상적으로 종료되려면 IndexError나 StopIteration을 출력해야 함
        # 그러나 Pandas Column을 그대로 사용하면 대신 KeyError가 나서 비정상 종료

        self.transform = transform

        self.idx = 0


    def __getitem__(self, index):
        '''
        index: 데이터 로더 등이 요구하는 파일의 번호
        '''

        # 현재 csv 파일에서는 데이터셋 최상위 폴더에 대한 상대경로가 저장되어 있으므로, 
        # 데이터셋 최상위 폴더를 덧붙여줌
        # print(index)
        # if index >= 13492:
        #     return None, None
        # try:
        #     file_path = os.path.join("./US_license_plates_dataset", self.file_list_by_csv[index])
        # except KeyError:
        #     raise IndexError

        file_path = os.path.join("./US_license_plates_dataset", self.file_list_by_csv[index])

        img = cv2.imread(file_path)  # 요구하는 파일을 cv2 이미지로 로딩
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2로 불러온 이미지는 BGR 채널이므로, RGB로 변환 필요

        # csv로 데이터를 구성할 경우, class id가 이미 명시되어 있으므로 그대로 반환값으로 사용
        label = self.class_id[index]

        if self.transform is not None:
            img = self.transform(image=img)['image'] # albumentation transform 사용시

        self.idx += 1
        return img, label

    def __len__(self):
        return len(self.file_list_by_csv)
    

if __name__ == "__main__":
    # 이번 파일을 테스트하기 위해 실행시 작동하는 코드
    dataset = SportsDataset("./US_license_plates_dataset/sports.csv")

    for item in dataset:
        _, label = item
        # print(label)