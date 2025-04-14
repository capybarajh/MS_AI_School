import torch
import torchvision
import matplotlib.pyplot as plt

from nosie_auto_encoder_model import DenoisingAutoEncoder

# transforms
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,),)
])
# model loader
model = DenoisingAutoEncoder()
model.load_state_dict(torch.load("./denoisingAutoEncoder_model.pt", map_location='cpu'))
model.eval()

# batch_size
batch_size = 10

# dataset, dataloader
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for images, _ in test_loader :

    # noise add
    # noise_factor = 0.2 ~ 0.5
    noise_factor = 0.2
    noisy_images = images + noise_factor * torch.randn(images.size())

    # model test
    # with torch.no.grad() :
    reconstructed_images = model(noisy_images)

    for j in range(batch_size) :
        print(reconstructed_images[j])

        fig, axes = plt.subplots(1, 3, figsize=(15,5))

        # org image
        original_images = images[j].view(28,28)
        axes[0].imshow(original_images, cmap='gray')
        axes[0].set_title('Original Image')

        # Noisy Image
        noisy_image = noisy_images[j].view(28,28)
        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title('Noisy Image')

        reconstructed_image = reconstructed_images[j].view(28,28)
        axes[2].imshow(reconstructed_image.detach(), cmap='gray')
        axes[2].set_title("Reconstructed Image")

        for ax in axes:
            ax.axis('off')

        plt.show()








