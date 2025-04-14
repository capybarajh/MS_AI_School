import os
import torch
import torchvision.transforms as transforms
from base_dcgan_model import BaseDcganGenerator
from PIL import Image

nz = 100

output_images_path = "./generated_images_256/"
os.makedirs(output_images_path, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

netG = BaseDcganGenerator().to(device)
checkpoint_path = "./DCGAN_model_weight/netG_epoch_195.pth"
netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
netG.eval()

# G image
num_images = 10
generated_images = []

with torch.no_grad():
    for _ in range(num_images) :
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake = netG(noise)
        generated_images.append(fake.detach().cpu())

for i, image in enumerate(generated_images) :
    image = torch.clamp(image, min=-1, max=1)
    image = (image + 1) / 2
    image = transforms.ToPILImage()(image.squeeze())
    image = image.resize((256,256), Image.BICUBIC)
    image.save(f"{output_images_path}generate_image_{i}.png")
