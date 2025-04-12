import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms

image_path = "./surprised_cat.jpg"
image = Image.open(image_path).convert('L')
input_data = transforms.ToTensor()(image).unsqueeze(0)

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2)

output = conv(input_data)

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("input image")

plt.subplot(1, 2, 2)
plt.imshow(output.squeeze().detach().numpy(), cmap='gray')
plt.title("output image")

plt.tight_layout()
plt.show()