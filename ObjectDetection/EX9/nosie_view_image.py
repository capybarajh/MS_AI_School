import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

batch_size = 128

# transforms
transforms = transforms.Compose([
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),

])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

def add_noise(image, noise_factor=0.5) :
    # noise_factor -> 잡음의 정도를 결정 하는 인자
    noisy_images = image + noise_factor * torch.rand_like(image)
    noisy_images = torch.clamp(noisy_images, -1, 1)

    return noisy_images

for images, _ in train_loader :
    print(images.shape)

    noisy_images = add_noise(images)
    break

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.transpose(torchvision.utils.make_grid(images[:8], padding=2, normalize=True), (1,2,0)))
axes[0].set_title('Original Images')
axes[0].axis('off')

axes[1].imshow(np.transpose(torchvision.utils.make_grid(noisy_images[:8], padding=2, normalize=True), (1,2,0)))
"""
RGB 이미지의 경우는 일반적으로 높이 너비 채널 순가지고요 
그레이 높이 너비 즉 높이, 너비, 채널 형식의 이미지를 시각화할때 (1, 2, 0),오르 np.transpose 사용하여 차원을 재배열 합니다. 
"""
axes[1].set_title('Noisy Images')
axes[1].axis('off')

plt.show()