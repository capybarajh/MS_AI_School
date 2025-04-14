import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

latent_dim = 20

latent_vector = torch.randn(1, latent_dim)

class Decoder(nn.Module) :
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 784)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

decoder = Decoder()

# tanh
decoder_tanh = decoder(latent_vector)
decoder_tanh = decoder_tanh.view(28,28).detach().numpy()

# Relu
decoder.tanh = nn.ReLU() # self.tanh = nn.Tanh() -> ReLU()
decoder_relu = decoder(latent_vector)
decoder_relu = decoder_relu.view(28, 28).detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].imshow(decoder_tanh, cmap='gray')
axes[0].set_title('Decoded with Tanh')
axes[0].axis('off')

axes[1].imshow(decoder_relu, cmap='gray')
axes[1].set_title('Decoded with relu')
axes[1].axis('off')

plt.show()