# 오토인코더 (MNIST) 예제를 통한 오토인코더의 학습
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from auto_encoder_model import Autoencoder

# hyperparameter
batch_size = 326
lr = 0.001
num_epochs = 100

# transforms
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2, pin_memory=True)

# model load
autoencoder = Autoencoder().to("cuda")
criterion = nn.MSELoss().to("cuda")
optimizer = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-4)

if __name__ == '__main__':

    # train
    for epoch in range(num_epochs) :
        start_time = time.time()

        for data in train_loader :
            img, _ = data
            img = img.to("cuda")
            img = img.view(img.size(0), -1).to('cuda') # 2D -> 1D

            optimizer.zero_grad()
            outputs = autoencoder(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"EPOCH [{epoch + 1} / {num_epochs}],"
              f" Loss : {loss.item():.4f} end tiem : {epoch_time:.2f} seconds")

    torch.save(autoencoder.state_dict(), 'autoencoder_model.pt')






