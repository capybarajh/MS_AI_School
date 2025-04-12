import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



def train(model, train_dataset_len, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        # 모델을 학습 모드로 설정
        running_loss = 0.0
        # 트레인 로더를 한번 순회한 오차값
        # start_time = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # loader에서 받아온 image와 label을 적절한 장치에 넘김
            optimizer.zero_grad() # 그래디언트 초기화

            outputs = model(images)
            # 모델에 받아온 images를 입력 후 예측값인 outputs를 받아옴

            loss = criterion(outputs, labels)
            # 실제 정답인 labels와 상기 예측값인 outputs를 손실함수로 대조하여 오차 출력

            loss.backward()
            optimizer.step()
            # 역전파 이후 optimizer가 가중치 업데이트

            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / train_dataset_len

        print(f"Epoch {epoch+1} / {num_epochs}, Loss : {epoch_loss:.4f}")
        # Epoch 순회마다 발생한 loss 출력


def eval(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f} %")