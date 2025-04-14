import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from torchvision.models.mobilenetv2 import mobilenet_v2
from albumentations.pytorch import ToTensorV2
from ex_01_0717_customdataset import MyFoodDataset


def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model setting
    # model = resnet50(pretrained=False)
    # model.fc = nn.Linear(2048, 20)

    model = mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(1280, 20)


    model.load_state_dict(torch.load(f="./ex01_0717_mobilenet_v2_best.pt"))

    # aug
    val_transforms = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])

    test_dataset = MyFoodDataset("./food_dataset/test/", transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()

    from tqdm import tqdm
    correct = 0
    with torch.no_grad() :
        for data, target in tqdm(test_loader) :
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()


    print("test set : Acc {}/{} [{:.0f}]%\n".format(
        correct, len(test_loader.dataset),
        100*correct / len(test_loader.dataset)
    ))




if __name__ == "__main__" :
    main()