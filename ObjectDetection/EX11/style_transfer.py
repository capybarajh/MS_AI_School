import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# output size
imsize = 512 if torch.cuda.is_available() else 128

loader_transforms = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def image_loader(image_name) :
    image = Image.open(image_name)
    image = loader_transforms(image).unsqueeze(0)
    return image.to(device, torch.float)

# style image content image loader
style_image = image_loader("./neural_style_transfer.jpg")
content_image = image_loader("./cat_input.jpg")

unloader = transforms.ToPILImage()

def imshow(tensor, title=None) :
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

plt.figure()
imshow(style_image, title='Style Image')

plt.figure()
imshow(content_image, title='Content Image')
plt.show()

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

import torch.nn as nn
class Normalization(nn.Module) :
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def foward(self, img):
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
