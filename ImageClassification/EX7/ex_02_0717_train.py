import torch.nn as nn
import torch
import torchivision
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from torchvision.models.efficientnet import efficientnet_b0
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from ex_02_0717_customdataset import MyDataset

def train(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []





def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = efficientnet_b0(pretrained=True)
    model.classifier = nn.Linear(1280, 10)
    model.to(device)

    # aug
    train_transforms = A.Compose([
        A.Resize(width=225, height=225),
        A.RandomShadow(),
        A.RandomBrightnessContrast(),
        A.HorizeontalFlip(),
        A.VerticalFlip(),
        ToTensorV2()         _
    ])

    val_transforms = A.Compose([
        A.Resize(width=255, height=255),
        ToTensorV2()
    ])

    train_dataset = MyDataset("./ex02_dataset/train/", transform=train_transforms)
    val_dataset = MyDataset("./ex02_dataset/val/", transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=124, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=124, shuffle=True)

    epochs = 40
    criterion = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)


if __name__ == "__main__":
    main()