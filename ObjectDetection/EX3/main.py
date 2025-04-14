import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from SegDataset import customVOCSegmentation
from argparse import ArgumentParser
from learning.learner import SegLearner


if __name__=="__main__":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # 실험에 의해 도출된, 가장 좋은 결과를 도출하는 평균/표준편차 값들

    train_transforms = A.Compose([
        A.Resize(520, 520), # DeepLabV3 pretrain 모델이 사용할 이미지 크기 (520x520)

        A.HorizontalFlip(p=0.5), # 50% 확률로 좌우반전

        A.Normalize(mean=mean, std=std),
        ToTensorV2() # torch 모델에 넣을 수 있도록 ToTensor 사용
    ])

    valid_transforms = A.Compose([
        A.Resize(520, 520), # DeepLabV3 pretrain 모델이 사용할 이미지 크기 (520x520)
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    # 아래에 dataset과 dataloader 등등에 필요한 터미널 인자를 지정하기 위해 argparser 사용
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data",
                        help="데이터셋 파일이 저장되거나 로딩될 지점")
    parser.add_argument("--weight_foler_path", type=str, default="./weight",
                        help="가중치가 저장될 폴더의 경로")
    parser.add_argument("--weight_file_name", type=str, default="weight.pt",
                        help="저장될 가중치 파일의 이름")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="데이터로더가 사용할 프로세스 수, 사용자 컴퓨터 환경에 따라 달라질 수 있음")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="lr의 감소율")
    parser.add_argument("--resume", action="store_true",
                        help="학습 재개 여부, store_true가 지정되면 터미널 인자로 선언되어야 true가 들어옴")
    parser.add_argument("--epochs", type=int, default=10,
                        help="이번 학습의 총 epoch 수")
    args = parser.parse_args()

    train_dataset = customVOCSegmentation(args.data_path, mode="train", transforms=train_transforms)
    valid_dataset = customVOCSegmentation(args.data_path, mode="val", transforms=valid_transforms)
    # dataset 선언

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    # dataloader 선언

    model = deeplabv3_resnet101(pretrained=True)
    # model 선언 (pretrained 모델 사용)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss(ignore_index=255)

    # train을 위해 만든 편의 class 선언
    learner = SegLearner(model, optimizer, criterion, train_dataloader, valid_dataloader, args)

    if args.resume:
        learner.load_ckpts()

    # 학습 시작
    learner.train()
    