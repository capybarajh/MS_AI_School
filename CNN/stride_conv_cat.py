import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

image_path = "./surprised_cat.jpg"
input_image = Image.open(image_path).convert('L')
# 이미지 읽기

# input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
input_data = torch.unsqueeze(torch.from_numpy(np.array(input_image)), dim=0).float()
# input_data = torch.unsqueeze(torch.from_numpy(input_image), dim=0).float()
# 이미지를 텐서로 변환

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
# Convolution layer 생성 (stride=2)
output = conv(input_data)

plt.subplot(1, 2, 1)
plt.imshow(input_data.squeeze().detach().numpy(), cmap='gray')
plt.title("Input image")

plt.subplot(1, 2, 2)
plt.imshow(output.squeeze().detach().numpy(), cmap='gray')
plt.title("Output")

plt.tight_layout()
plt.show()
