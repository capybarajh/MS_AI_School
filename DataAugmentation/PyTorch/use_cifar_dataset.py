import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt

def imgaug_transform(image: torch.Tensor):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.Multiply((0.8, 1.2))
    ])
    image_np = image.permute(1, 2, 0).numpy()
    image_aug = seq(image=image_np)
    image_aug_copy = image_aug.copy()
    image_aug_tensor = torch.from_numpy(image_aug_copy).permute(2, 0, 1)
    return image_aug_tensor

def transform_data(image):
    tensor = transforms.ToTensor()(image)
    transformed_tensor = imgaug_transform(tensor)
    return transformed_tensor

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_data)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for images, labels in train_dataloader:
    fig, axes = plt.subplots(1, batch_size, figsize=(12, 4))

    for i in range(batch_size):
        image = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(image)
        axes[i].set_title(f"Label : {labels[i]}")

    plt.show()
    break
