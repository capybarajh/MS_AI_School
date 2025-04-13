# VGG11모델 과  ResNet18모델 앙상블 실습
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg11, resnet18

from sklearn.metrics import accuracy_score

# GPU setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 전처리
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터 정의
train_dataset = CIFAR10(root="./data", train=True, download=False,
                        transform=train_transform)
test_dataset = CIFAR10(root="./data", train=False, download=False,
                       transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=246, shuffle=True, num_workers=2,
                          pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=246, shuffle=False, num_workers=2,
                         pin_memory=True)

# vgg11 모델 과 resnet18 모델 정의
vgg_model = vgg11(pretrained=True)
resnet_model = resnet18(pretrained=True)

num_features_vgg = vgg_model.classifier[6].in_features
num_features_resnet = resnet_model.fc.in_features

vgg_model.classifier[6] = nn.Linear(num_features_vgg, 10)
resnet_model.fc = nn.Linear(num_features_resnet, 10)

# Voting 앙상블 모델 정의
class EnsembleModel(nn.Module) :
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        avg_output = torch.mean(outputs, dim=0)

        return avg_output

ensemble_model = EnsembleModel([vgg_model, resnet_model])
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(ensemble_model.parameters(), lr=0.001)

# 모델 학습 함수 정의
def train(model, device, train_loader , optimizer, criterion) :
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader) :
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, device, test_loader) :
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad() :
        for data, target in test_loader :
            data, target = data.to(device) , target.to(device)
            output = model(data)
            _, predicted = torch.max(output,1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(target.cpu().numpy())

    acc = accuracy_score(targets, predictions)
    return acc

# 앙상블된 모델 예측 합치는 함수 정의
def combine_predictions(predictions) :
    combined = torch.cat(predictions, dim=0)
    _, predicted_lables = torch.max(combined, 1)
    return predicted_lables

if __name__ == '__main__' :
    for epoch in range(1, 20) :
        print(f"Train model ... {epoch}")
        ensemble_model = ensemble_model.to(device)
        train(ensemble_model, device, train_loader, optimizer, criterion)
        predictions = []
        with torch.no_grad() :
            for data, _ in test_loader  :
                data = data.to(device)
                output = ensemble_model(data)
                predictions.append(output)

        combine_predictions_ = combine_predictions(predictions)
        acc = accuracy_score(test_dataset.targets, combine_predictions_.cpu().numpy())

        print(f"epoch >> [{epoch}] acc >> [{acc:.2f}]")