import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from vae_model_0814 import VAE

batch_size = 246
learning_rate = 0.0025
latent_dim = 20
num_epochs = 150

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

# train_dataset, loader
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)

# model load
model = VAE()

# optimizer loss
criterion = nn.BCELoss(reduction='sum') #  매개변수는 손실을 계산할 때 어떤 방식으로 감소할지를 나타냅니다.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train loop
for epoch in range(num_epochs) :
    for i, (images, _) in enumerate(train_loader) :
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()

        recon_images, mu, logval = model(images)

        # 재구성 오차
        # recon_images : 디코더를 통해 재구성된 이미지
        # images : input images
        # 재구성 오차 -> 입력 이미지와 재구성된 이미지 간 차이 계산하고 배치 크기로 나눠 정규화
        reconstruction_loss = criterion(recon_images, images) / batch_size

        # KL
        # KL 다이버전스 계산식은 잠재 변수의 분포와 정규 분포 사이의 차이를 측정 / 이것은 잠재 변수의 분포가 얼마나 정규 분포와 유사한지를 나타낸다.
        # mu : 잠재 변수의 평균
        # logval : 잠재 변수의 로그 분산 값
        kl_divergence = -0.5 * torch.sum(1 + logval - mu.pow(2) - logval.exp()) / batch_size

        # all loss
        loss = reconstruction_loss + kl_divergence

        # VAE의 목표는 재구성 오차를 최소화 하면서 동시에 KL 다이버전스 줄이는 것입니다.

        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss : {loss.item():.4f}")

torch.save(model.state_dict(), "./vae_model.pth")