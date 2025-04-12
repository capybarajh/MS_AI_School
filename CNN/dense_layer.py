import torch
import torch.nn as nn
import matplotlib.pyplot as plt

input_size = 4
output_size = 2
# 입력 크기 4, 출력 크기 2 (임의 지정)

dense_layer = nn.Linear(input_size, output_size)
# 밀집층 정의

weights = dense_layer.weight.detach().numpy()

plt.imshow(weights, cmap='coolwarm', aspect='auto')
plt.xlabel("input features") # 입력 요소 (4개)
plt.ylabel("Output units") # 출력 요소 (2개)
plt.title("Dense Layer Weights")
plt.colorbar()
plt.show()