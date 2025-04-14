import torch
import matplotlib.pyplot as plt
from auto_encoder_model import Autoencoder


# model load
load_auto_encoder = Autoencoder()
load_auto_encoder.load_state_dict(torch.load("./autoencoder_model.pt", map_location="cpu"))
load_auto_encoder.eval()

with torch.no_grad() :
    test_sample = torch.randn(1,32)
    print(test_sample)
    generated_sample = load_auto_encoder.decoder(test_sample).view(1,1,28,28)

plt.imshow(generated_sample.squeeze().numpy(), cmap='gray')
plt.show()