import torch
import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class KeypointDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):
        self.demo = demo
        # Visualize를 통해 시각화로 확인하고자 할때 비교를 위한 변수
        self.root = root
        # 현재 데이터셋은 root 인자에 dataset 하위 폴더의 train, test 폴더를 택일하도록 되어 있음
        self.imgs_files = sorted(os.listdir(os.path.join(self.root, "images")))
        # 이미지 파일 리스트. train 또는 test 폴더 하위에 있는 images 폴더를 지정하고, 해당 폴더의 내용물을 받아옴
        # 이미지를 이름 정렬순으로 불러오도록 sorted를 붙임
        self.annotations_files = sorted(os.listdir(os.path.join(self.root, "annotations")))
        # 라벨링 JSON 파일 리스트. 상기 images 폴더를 받아온 것과 동일
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        # 이번에 호출되는 idx번째 이미지 파일의 절대경로
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])
        # 이번에 호출되는 idx번째 이미지 파일의 라벨 JSON 파일 경로

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        # 이미지를 읽은 후, BGR 순서를 RGB 형태로 바꿈

        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 라벨 JSON 파일을 JSON 모듈로 받아옴
            bboxes_original = data["bboxes"]
            # JSON 하위 "bboxes" 키로 bbox 정보들이 담겨있음
            keypoints_original = data["keypoints"]
            # "keypoints" 키로 키포인트 정보가 담겨있음
            bboxes_labels_original = ['Glue tube' for _ in bboxes_original]
            # 현재 데이터셋은 모든 객체가 접착제 튜브이므로, 
            # 모든 bbox에 대해 일관적으로 'Glue tube'라는 라벨을 붙여줌
        
        if self.transform: # if self.transform is not None:
            # "keypoints": [
            #   [[1019, 487, 1], [1432, 404, 1]], [[861, 534, 1], [392, 666, 1]]
            # ]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            # kp : [[1019, 487, 1]]
            # el : [1019, 487, 1]
            # el[0:2] : [1019, 487] (평면 이미지에서 3번째 축 요소는 필요없기 때문에 제거)
            # keypoints_original_flattened = [[1019, 487], [1432, 404], [861, 534], [392, 666]]
            # albumentation transform은 평면 이미지에 적용되므로 2차원 좌표만이 필요함

            # albumentation 적용
            transformed = self.transform(image=img_original, bboxes=bboxes_original,
                                         bboxes_labels=bboxes_labels_original,
                                         keypoints=keypoints_original_flattened)
            
            
            img = transformed["image"] # albumentation transform이 적용된 image
            bboxes = transformed["bboxes"]

            keypoints_transformed_unflattened = np.reshape(np.array(transformed["keypoints"]), (-1, 2, 2)).tolist()
            # transformed["keypoints"] : [1019, 487, 1432, 404, 861, 534, 392, 666]
            # keypoints_transformed_unflattened : [[[1019, 487], [1432, 404]], [[861, 534], [392, 666]]]

            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened):
                obj_keypoints = []
                # o_idx : 현재 순회중인 요소의 순번 (index)
                # obj : 현재 순회중인 요소, ex) [[1019, 487], [1432, 404]]
                for k_idx, kp in enumerate(obj):
                    # k_idx : 현재 순회중인 하위 요소의 순번 (index)
                    # kp : 현재 순회중인 요소, ex) [1019, 487]
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                    # torch.Tensor에서 벡터곱을 하는 과정에서 필요할 3번째 축 요소를 덧붙임 ex) [1019, 487, 1]
                keypoints.append(obj_keypoints)
                # Tensor 형태로 사용할 keypoints 리스트에 3번째 축 요소를 덧붙인 키포인트 좌표를 담음

        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original
            # transform이 없는 경우에는 변수 이름만 바꿔줌
        

        # transform을 통과한 값들을 모두 tensor로 변경
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # as_tensor 메서드가 list를 tensor로 변환할 때 속도 이점이 있음
        target = {}
        # keypoint 모델에 사용하기 위한 label이 dictionary 형태로 필요하므로, dict 형태로 꾸림
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64)
        # 모든 객체는 동일하게 접착제 튜브이므로, 동일한 라벨 번호 삽입
        target["image_id"] = torch.tensor([idx])
        # image_id는 고유번호를 지칭하는 경우도 있는데, 그러한 경우에는 JSON 파일에 기입이 되어있어야 함
        # 이번 데이터셋은 JSON상에 기입되어있지 않으므로, 현재 파일의 순번을 넣어줌
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        # 해당하는 bbox의 넓이. 
        # bboxes[:, 3] - bboxes[:, 1] : bboxes 내부 요소들의 (y2 - y1), 즉 세로 길이
        # bboxes[:, 2] - bboxes[:, 0] : bboxes 내부 요소들의 (x2 - x1), 즉 가로 길이
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        # 이미지상에 키포인트 또는 bbox가 가려져있는지를 묻는 요소
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)

        img = F.to_tensor(img)
        # image의 텐서 변환

        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original],
                                                    dtype=torch.int64)  # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (
                    bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)
        img_original = F.to_tensor(img_original)
        # demo=True일 경우, 원본 이미지와 변환된 이미지를 비교하기 위해 원본 이미지를 반환하기 위한 블록

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target
    

    def __len__(self):
        return len(self.imgs_files)


# if __name__=="__main__":
#     root_path = "./keypoint_dataset"
#     train_dataset = KeypointDataset(f"{root_path}/train")
#     for item in train_dataset:
#         print(item)