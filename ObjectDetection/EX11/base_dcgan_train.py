import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dast
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from base_dcgan_model import BaseDcganGenerator, BaseDcganDiscriminator
from base_dcgan_config import *

def weights_init(m) : # 모델 함수 초기화
    classname = m.__class__.__name__
    # Conv 라는 문자열이 포함되어 있는 경우 -> 가중치 평균 0 표준편차가 0.02 인 정규 분포에서 무작위로 초기화 즉 정규분포를 따르는 난수로 가중치 초기화
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    # BatchNorm 문자열이 포함되어있는 경우 -> 가중치 평균 1, 표준편차가 0.02 에서 무작위로 초기화 편향 0 으로 설정
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, val=0)
    # 일반적인 딥러닝 모델의 초기화 단계에서 사용되는 거고 모델 학습, 수렴 속도 개선에 도움을 줍니다.

def main() :
    # transform
    data_transform = transforms.Compose([
                                   transforms.Resize(img_size),
                                   # transforms.RandAugment(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
                               ])

    # dataset
    dataset = dast.ImageFolder(root=data_root, transform=data_transform)
    # GAN -> image -> custom dataset -> return image

    # dataloader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # data loader view
    real_batch = next(iter(data_loader))
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title('Training data Image view')
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    # train loop
    output_images_path = "./generated_images/"
    os.makedirs(output_images_path, exist_ok=True)

    # loss function
    criterion = nn.BCELoss()
    # Generator input noise
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Generator model load
    netG = BaseDcganGenerator().to(device)
    # DCGAN -> mean = 0 stdev = 0.02
    netG.apply(weights_init)

    # Discriminator model load
    netD = BaseDcganDiscriminator().to(device)
    # DCGAN -> mean = 0 stdev = 0.02
    netD.apply(weights_init)

    # true false -> Discriminator
    real_label = 1.
    fake_label = 0.

    # Generator, Discriminator optimizer
    optimizerD = optim.AdamW(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.AdamW(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop .... ")

    for epoch in range(num_epochs) :
        for i, data in enumerate(data_loader, 0) :
            ############################
            # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
            ###########################
            netD.zero_grad() # 판별자 그래디언트 초기화 -> 배치 마다 새로운 그래디언트 계산하기 위해서

            real_gpu = data[0].to(device) # 실제 데이터를 -> GPU 혹은 CPU 올립니다.
            b_size = real_gpu.size(0) # 현재 배치의 크기 (128, 3, 64,64)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # 실제 데이터에 대한 라벨을 생성 -> 판별자 실제 혹은 가짜 인지 알려주는 역활 -> 해당 하는 예상되는 값

            output = netD(real_gpu).view(-1) # 실제 데이터 -> 판별 모델 -> 출력 (출력형태가 1차원 백터로 변환)

            loss_d = criterion(output, label) # # 판별자 출력 과 실제 데이터 간의 라벨 차이 계산
            loss_d.backward() # 역잔파 해서 가중치 업데이트 -> 잘 구별할 수 있도록 학습 유도

            D_x = output.mean().item()

            # fack images
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)

            # D를 이용해 데이터의 진위를 판별합니다
            output = netD(fake_images.detach()).view(-1)

            # D loss
            loss_d_fake = criterion(output, label)
            loss_d_fake.backward()

            D_G_z1 = output.mean().item()

            errD = loss_d + loss_d_fake
            optimizerD.step()

            ############################
            # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
            # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
            # 이때 G는 업데이트되지 않았지만, D가 업데이트 되었기 때문에 앞선 손실값가 다른 값이 나오게 됩니다
            output = netD(fake_images).view(-1)

            # G loss
            loss_g = criterion(output, label)

            loss_g.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            # 훈련 상태를 출력합니다
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(data_loader),
                         errD.item(), loss_g.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(loss_g.item())
            D_losses.append(errD.item())

            # fixed noise -> 6 images append
            if (iters % 500 == 0 ) or ((epoch == num_epochs -1) and (i == len(data_loader)-1)) :
                with torch.no_grad() :
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            # model save
            if epoch % 5 == 0 :
                os.makedirs("./DCGAN_model_weight/", exist_ok=True)
                torch.save(netG.state_dict(), f"./DCGAN_model_weight/netG_epoch_{epoch}.pth")
                torch.save(netD.state_dict(), f"./DCGAN_model_weight/netD_epoch_{epoch}.pth")

            # epoch 5
            if (epoch + 1) % 5 == 0:
                vutils.save_image(img_list[-1], f"{output_images_path}/fake_image_epoch_{epoch}.png")

            iters +=1




if __name__ == '__main__' :
    main()
