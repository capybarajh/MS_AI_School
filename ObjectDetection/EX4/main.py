import torch
import torchvision
import albumentations as A

from engine import train_one_epoch
from utils import collate_fn

from torch.utils.data import DataLoader
from Customdataset import KeypointDataset

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import keypointrcnn_resnet50_fpn


def get_model(num_keypoints, weights_path=None):
    # 필요한 모델에 대해 키포인트 개수를 정의하고, 기존 모델이 있는 경우 로드하는 편의함수

    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), 
                                       aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    # 여러 input size에 대해 feature map으로 넘어갈 때 적절한 비율 변환을 도와주는 객체

    model = keypointrcnn_resnet50_fpn(pretrained=False, # 모델 자체는 pretrained 되지 않은 모델
                                      # 현재는 학습용 코드이기 때문에, pretrain 모델을 fine-tuning하지 않고
                                      # 처음부터 가중치 업데이트를 하도록 설정
                                      pretrained_backbone=True, # backbone은 pretrained 된 모델
                                      # backbone은 모델 설계상 이미 pretrain 됐을 것으로 상정했기 때문에
                                      # 실제 가중치 업데이트가 주로 일어날 부분에 비해 pretrain 여부가 크게 상관있지 않음
                                      num_keypoints=num_keypoints,
                                      num_classes=2, # 무조건 배경 클래스를 포함함
                                      rpn_anchor_generator=anchor_generator)
    
    if weights_path: # 기존 저장된 모델 가중치를 사용할 때
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model
    

train_transform = A.Compose([
    A.Sequential([
        A.RandomRotate90(p=1),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True,
                                   always_apply=False, p=1)
    ], p=1)
], keypoint_params=A.KeypointParams(format="xy"),
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=['bboxes_labels']))
# train dataset에 사용할 Albumentation transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KEYPOINTS_FOLDER_TRAIN = "./keypoint_dataset/train/"
# train dataset이 있는 경로
train_dataset = KeypointDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)

model = get_model(num_keypoints=2)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
# momentum (==관성) : 이전에 가중치를 업데이트한 기울기를 얼마나 반영할 것인가?
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
# StepLR : step_size epoch마다 lr에 gamma가 곱해짐
num_epochs = 20
# 최대 학습 횟수

# 현재 engine에서 받아온 train_one_epoch 함수는 손실함수를 넣지 않아도 되도록 처리된 함수이므로,
# 손실함수의 정의는 생략됨

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=1000)
    # 학습이 진행되는 함수
    lr_scheduler.step()
    # 학습 1 epoch가 끝날때마다 scheduler 역시 업데이트
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"./keypointsrcnn_weights_{epoch}.pth")
        # 10 epochs마다 가중치 저장

torch.save(model.state_dict(), "./keypointsrcnn_weights_last.pth")
# 모든 epoch가 끝나면 last.pth까지 저장
