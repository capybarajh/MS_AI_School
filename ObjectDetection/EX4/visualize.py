import cv2
import albumentations as A
from Customdataset import KeypointDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from utils import collate_fn


train_transform = A.Compose([
                    A.Sequential([
                        A.RandomRotate90(p=1), # 랜덤 90도 회전
                        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True,
                                                   always_apply=False, p=1) # 랜덤 밝기 및 대비 조정
                    ], p=1)
                ], keypoint_params=A.KeypointParams(format='xy'),
                   bbox_params=A.BboxParams(format="pascal_voc", label_fields=['bboxes_labels']) # 키포인트의 형태를 x-y 순으로 지정
                )

root_path = "./keypoint_dataset/"
dataset = KeypointDataset(f"{root_path}/train/", transform=train_transform, demo=True)
# demo=True 인자를 먹으면 Dataset이 변환된 이미지에 더해 transform 이전 이미지까지 반환하도록 지정됨
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

iterator = iter(data_loader)
# for문으로 돌릴 수 있는 복합 자료형들은 iterable (반복 가능) 속성을 갖고 있음
# iter()로 그러한 자료형을 감싸면 iterator (반복자) 가 되고,
# next(iterator)를 호출하면서 for문을 돌리듯이 내부 값들을 순회할 수 있게 됨
batch = next(iterator)
# iterator에 대해 next로 감싸서 호출을 하게 되면, 
# for item in iterator의 예시에서 item에 해당하는 단일 항목을 반환함
# 아래 4줄에 해당하는 코드와 같은 의미
# batch_ = None
# for item in data_loader:
#     batch_ = item
#     break

keypoints_classes_ids2names = {0: "Head", 1: "Tail"}
# bbox 클래스는 모두 접착제 튜브 (Glue tube) 로 동일하지만, keypoint 클래스는 위의 dict를 따름


def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    # pyplot을 통해서 bbox와 키포인트가 포함된
    # 원본 이미지와 변환된 이미지를 차트에 띄워서 대조할 수 있는 편의함수
    fontsize = 18
    # cv2.putText에 사용될 글씨 크기 변수

    for bbox in bboxes:
        # bbox = xyxy
        start_point = (bbox[0], bbox[1])
        # 사각형의 좌측 상단
        end_point = (bbox[2], bbox[3])
        # 사각형의 우측 하단
        image = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 2)
        # 이미지에 bbox 좌표에 해당하는 사각형을 그림

    for kpts in keypoints:
        # keypoints : JSON 파일에 있는 keypoints 키의 값 (즉, keypoints 최상위 리스트)
        # kpts : keypoints 내부의 각 리스트 (즉, 각 bbox의 키포인트 리스트)
        for idx, kp in enumerate(kpts):
            # kp : kpts 내부의 각 리스트 (즉, 키포인트 리스트 내부의 xy좌표쌍, 키포인트 점)
            image = cv2.circle(image.copy(), tuple(kp), 5, (255, 0, 0), 10)
            # 현재 키포인트에 점을 찍음
            image = cv2.putText(image.copy(), f" {keypoints_classes_ids2names[idx]}", tuple(kp),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
            # 현재 키포인트가 Head인지 Tail인지 위에 선언한 dict에 해당하는 문자를 집어넣음

    # 변환된 이미지만을 확인할 경우, 원본 이미지가 없을 것이므로 그대로 이미지 처리 끝냄
    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40, 40))
        # 이미지를 그릴 차트 선언
        plt.imshow(image)
        # 위에서 bbox와 키포인트를 그린 이미지를 출력

    else:
        for bbox in bboxes_original:
        # bbox = xyxy
            start_point = (bbox[0], bbox[1])
            # 사각형의 좌측 상단
            end_point = (bbox[2], bbox[3])
            # 사각형의 우측 하단
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0, 255, 0), 2)
            # 이미지에 bbox 좌표에 해당하는 사각형을 그림

        for kpts in keypoints_original:
            # keypoints : JSON 파일에 있는 keypoints 키의 값 (즉, keypoints 최상위 리스트)
            # kpts : keypoints 내부의 각 리스트 (즉, 각 bbox의 키포인트 리스트)
            for idx, kp in enumerate(kpts):
                # kp : kpts 내부의 각 리스트 (즉, 키포인트 리스트 내부의 xy좌표쌍, 키포인트 점)
                image_original = cv2.circle(image_original.copy(), tuple(kp), 5, (255, 0, 0), 10)
                # 현재 키포인트에 점을 찍음
                image_original = cv2.putText(image_original.copy(), f" {keypoints_classes_ids2names[idx]}", tuple(kp),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
                # 현재 키포인트가 Head인지 Tail인지 위에 선언한 dict에 해당하는 문자를 집어넣음
        
        f, ax = plt.subplots(1, 2, figsize=(40, 20))
        # 두 장의 이미지를 1행 2열, 즉 가로로 길게 보여주는 subplots 생성

        ax[0].imshow(image_original)
        # 첫번째 subplot에는 원본 이미지를 출력
        ax[0].set_title("Original Image", fontsize=fontsize)
        # 이미지 제목

        ax[1].imshow(image)
        # 두번째 subplot에는 변환이 완료된 이미지를 출력
        ax[1].set_title("Transformed Image", fontsize=fontsize)

        plt.show()


if __name__=="__main__":

    visualize_image_show = True
    visualize_targets_show = True

    image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # CustomDataset에서 Tensor로 변환했기 때문에 다시 plt에 사용할 수 있도록 numpy 행렬로 변경
    # img, target, img_original, target_original = batch이므로, batch[0]는 img를 지칭
    # batch[0][0]에 실제 이미지 행렬에 해당하는 텐서가 있을것 (batch[0][1]에는 dtype 등의 다른 정보가 있음)
    bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()
    # target['boxes']에 bbox 정보가 저장되어있으므로, 해당 키로 접근하여 bbox 정보를 획득

    keypoints = []
    for kpts in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints.append([kp[:2] for kp in kpts])
        # 이미지 평면상 점들이 필요하므로, 3번째 요소로 들어있을 1을 제거

    image_original = (batch[2][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # batch[2] : image_original
    bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()
    # batch[3] : target

    keypoints_original = []
    for kpts in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints_original.append([kp[:2] for kp in kpts])

    if visualize_image_show:
        visualize(image, bboxes, keypoints, image_original, bboxes_original, keypoints_original)
    if visualize_targets_show and visualize_image_show == False:
        print("Original targets: \n", batch[3], "\n\n")
        # original targets: (줄바꿈) original targets dict 출력 (두줄 내림)
        print("Transformed targets: \n", batch[1])