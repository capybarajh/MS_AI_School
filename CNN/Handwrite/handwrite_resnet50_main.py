from torchvision import datasets
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision import models

# 채널 별 mean 계산
def get_mean(dataset):
  meanRGB = [np.mean(image.numpy(), axis=(1,2)) for image,_ in dataset]
  meanR = np.mean([m[0] for m in meanRGB])
  meanG = np.mean([m[1] for m in meanRGB])
  meanB = np.mean([m[2] for m in meanRGB])
  return [meanR, meanG, meanB]

# 채널 별 str 계산
def get_std(dataset):
  stdRGB = [np.std(image.numpy(), axis=(1,2)) for image,_ in dataset]
  stdR = np.mean([s[0] for s in stdRGB])
  stdG = np.mean([s[1] for s in stdRGB])
  stdB = np.mean([s[2] for s in stdRGB])
  return [stdR, stdG, stdB]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # model = CNN().to(device)
    os.mkdir('./train')
    os.mkdir('./test')

    train_dataset = datasets.STL10('/train', split='train', download=True, transform=transforms.ToTensor())
    test_dataset = datasets.STL10('/test', split='test', download=True, transform=transforms.ToTensor())

    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(get_mean(train_dataset), get_std(train_dataset))])
    test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(get_mean(test_dataset), get_std(test_dataset))])

    # trainsform 정의
    train_dataset.transform = train_transforms
    test_dataset.transform = test_transforms

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = models.resnet50(pretrained=True).to(device)
    unpretrained_model = models.resnet50(pretrained=False)
    models.ResNet()
    
    lr = 0.0001
    num_epochs = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer)
    loss_function = nn.CrossEntropyLoss().to(device)

    params = {
    'num_epochs':num_epochs,
    'optimizer':optimizer,
    'loss_function':loss_function,
    'train_dataloader':train_dataloader,
    'test_dataloader': test_dataloader,
    'device':device
    }
    
    def train(model, params):
        loss_function=params["loss_function"]
        train_dataloader=params["train_dataloader"]
        test_dataloader=params["test_dataloader"]
        device=params["device"]

        for epoch in range(0, num_epochs):
            for i, data in enumerate(train_dataloader, 0):
                # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 이전 batch에서 계산된 가중치를 초기화
                optimizer.zero_grad() 

                # forward + back propagation 연산
                outputs = model(inputs)
                train_loss = loss_function(outputs, labels)
                train_loss.backward()
                optimizer.step()

            # test accuracy 계산
            total = 0
            correct = 0
            accuracy = []
            for i, data in enumerate(test_dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 결과값 연산
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss = loss_function(outputs, labels).item()
                accuracy.append(100 * correct/total)

            # 학습 결과 출력
            print('Epoch: %d/%d, Train loss: %.6f, Test loss: %.6f, Accuracy: %.2f' %(epoch+1, num_epochs, train_loss.item(), test_loss, 100*correct/total))


    train(model, params)