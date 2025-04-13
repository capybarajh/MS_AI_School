import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# GPU 설정 (사용 가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device >>" , device)

# 데이터셋 불러오기 .
train_transform = transform.Compose([
    transform.RandomHorizontalFlip(),
    transform.RandomVerticalFlip(),
    transform.RandAugment(),
    transform.ToTensor(),
    transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 이미지를 -1 ~ 1로 정규화
])

test_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 -1 ~ 1로 정규화
])

# 데이터셋, 데이터 로더
train_dataset = CIFAR10(root="./data", train=True, download=False,
                        transform=train_transform)
test_dataset = CIFAR10(root="./data", train=False, download=False,
                       transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# ResNet-18 모델 정의
model = resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10) # 클래스 개수 10개 입니다.
# print("fc in features >> ", num_features)

# 배깅 앙상블 모델 정의
"""
BaggingClassifier() => 여러개 분류기를 앙상블 해서 더 좋은 결과를 얻기 위한 알고리즘
DecisionTreeClassifier() => 결정나무 트리 알고리즘 (트리 깊이가 7로 제한)
"""
bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=7),
    n_estimators=5
)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 모델 학습 함수 정의
def train(model, device, train_loader, optimizer, criterion) :
    model.train()
    for batch_idx , (data, target) in enumerate(train_loader) :
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

# 모델 평가 함수 정의
def evalute(model, device, test_loader) :
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad() :
        for data, target in test_loader :
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output,1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    acc = accuracy_score(targets, predictions)

    return acc

# 앙상블된 모델 예측 함수 정의
def ensemble_pred(models, device, test_loader ) :
    predictions = []
    with torch.no_grad() :
        for data,_ in test_loader :
            data = data.to(device)
            outputs = []
            for model in models :
                model = model.to(device)
                model.eval()
                output = model(data)
                outputs.append(output)

            ensemble_output = torch.stack(outputs).mean(dim=0)
            _, pred = torch.max(ensemble_output, 1)
            predictions.extend(pred.cpu().numpy())

    return predictions

if __name__ == '__main__':

    models = []
    for epoch in range(1, 20) :
        print(f"Train ... {epoch}")
        model = model.to(device)
        train(model, device, train_loader, optimizer, criterion)
        acc = evalute(model, device, test_loader)
        print(f"Model {epoch} ACC {acc:.2f}")
        models.append(model)

    ensemble_predictions = ensemble_pred(models, device, test_loader)
    ensemble_acc = accuracy_score(test_dataset.targets, ensemble_predictions)
    print(f"\nEnsemble Acc : {ensemble_acc:.2f}")