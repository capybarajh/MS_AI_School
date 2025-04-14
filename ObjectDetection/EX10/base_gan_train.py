import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from base_gan_model_0814 import Generator, Discriminator

learning_rate = 0.0002
batch_size = 128
epochs = 400

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model load (generator, discriminator)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_G = optim.AdamW(generator.parameters(), lr=learning_rate)
optimizer_D = optim.AdamW(discriminator.parameters(), lr=learning_rate)

# transforms
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# data loader dataset
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# train loop
for epoch in range(epochs) :
    for i, (real_images, _) in enumerate(dataloader) :
        # 판별자의 그레이디언트 0으로 초기화 -> 역전파 단계에서 새로운 그레이디언트 계산하기전에 기존 값을 제거
        discriminator.zero_grad()
        batch_size = real_images.size(0) # 배치 사이즈 값 가져오기 (64, 1, 28, 28)
        real_labels = torch.ones(batch_size, 1).to(device) # 실제 이미지에 대한 레이블을 1로 설정하여 실제 이미지를 판별할 때 사용할 레이블을 생성
        fake_labels = torch.zeros(batch_size, 1).to(device) # 생성된 가짜 이미지에 대한 레이블을 0으로 설정하여 가짜 이미지를 판별할 때 사용할 레이블을 생성
        real_images = real_images.view(-1, 784).to(device)
        """
        실제 이미지 데이터를 모델에 입력하기 위해서 형태 조정 
            -> -1 (해당 차원의 크기를 유지하면서 다른 차원으로 크기를 조정 의미) 
                -> 2차원 -> 1차원 백터로 변환하여 모델 입력 준비 
        """

        # real images discriminator val, loss
        outputs_real = discriminator(real_images) # 실제 이미지 -> 판별자 모델  -> 판별 결과
        loss_real = criterion(outputs_real, real_labels)
        # 실제 이미지 -> 판별자의 출력 결과 -> 실제 이미지에 대한 레이블 간의 손실 계산 (판별자가 얼마나 실제 이미지를 정확하게 편별하는가를 나타내는 값 )

        # fake images discriminator val, loss
        noise = torch.randn(batch_size, 100).to(device)
        # 생성자에 입력으로 들어갈 노이즈 백터 생성 -> 노이즈 백터는 가짜 이미지 생성하는 사용 -> 100 : 노이즈 백터의 차원
        fake_images = generator(noise) # 노이즈 백터 ->  생성 모델 -> 가짜 이미지 생성
        fake_images = fake_images.view(-1, 784) # 생성된 가짜 이미지 데이터를 -> 판별기 모델 넣기 위해서 1차원 백터로 변환 필요
        outputs_fake = discriminator(fake_images.detach()) # 생성된 가짜 이미지 -> 판별 모델 -> 판별 결과 출력
        loss_fake = criterion(outputs_fake, fake_labels)
        """
        # 가짜 이미지에 대한 판별자 모델->출력결과->가짜 이미지에 대한 레이블 간의 손실 계산 
        (이 손실은 결국 판별자가 가짜 이미지를 얼마나 정확하게 판별하는가를 나타냅니다.)
        """
        # all loss (Discriminator)
        loss_D = loss_real + loss_fake # 판별자의 전체 손실 -> 실제 이미지에 대한 손실 과 가짜 이미지에 대한 손실 합친값
        loss_D.backward()
        optimizer_D.step()

        # generator train
        generator.zero_grad() # 생성자의 그레이디언트 0 초기화
        # 생성자는 판별자를 속이는 방향으로 학습 되어야합니다 -> 그래서 생성자만의 그레이디언트를 따로 계산 하고 업데이트 진행합니다.
        outputs_fake = discriminator(fake_images)
        loss_G = criterion(outputs_fake, real_labels) # 생성된 가짜 이미지에 대한 판별자의 출력 결과와 실제 이미지에 대한 레이블간 손실
        loss_G.backward()
        optimizer_G.step()

        # if i % 100 == 0 :
        print(f"Epoch [{epoch+1}/{epochs}], Step[{i+1}/{len(dataloader)}],"
              f"Discriminator Loss : {loss_D.item():.4f}, Generator Loss : {loss_G.item():.4f}")

    if (epoch + 1) % 10 == 0 :
        from torchvision.utils import save_image
        noise = torch.randn(1, 100).to(device)
        fake_image = generator(noise)
        fake_image = 0.5 * (fake_image + 1)
        fake_image = fake_image.copy()
        save_image(fake_image.view(1,1,28,28), f"generator_image_epoch_{epoch + 1}.png", normalize=True)

torch.save(generator.state_dict(), "generator_model.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")