import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from nosie_auto_encoder_model import DenoisingAutoEncoder

batch_size = 246
lr = 0.0025
num_epochs = 50
latent_dim = 20

transforms = transforms.Compose([
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) ,(0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

def add_noise(image, noise_factor=0.5) :
    # noise_factor -> 잡음의 정도를 결정 하는 인자
    noisy_images = image + noise_factor * torch.rand_like(image)
    noisy_images = torch.clamp(noisy_images, -1, 1)

    return noisy_images

model = DenoisingAutoEncoder().to('cuda')

criterion = nn.MSELoss().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs) :
    for i, (image, _) in enumerate(train_loader) :
        noisy_images = add_noise(image, noise_factor=0.3)
        image = image.to('cuda')
        noisy_images = noisy_images.to("cuda")

        optimizer.zero_grad()
        outputs = model(noisy_images)

        loss = criterion(outputs.view(-1, 784), image.view(-1, 784))
        loss.backward()
        optimizer.step()

    print(f"Epoch : [{epoch + 1 } / {num_epochs}] loss : {loss.item():.4f}")

torch.save(model.state_dict(), "./denoisingAutoEncoder_model.pt")


