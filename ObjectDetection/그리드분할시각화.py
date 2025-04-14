import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

image_path = "./image01.jpg"
image = Image.open(image_path)

# convert the image to a Pytorch tensor 
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).float()

# 그리드 생성 및 크기 조정 
grid_size = 20
heigth, width = image_tensor.shape[1], image_tensor.shape[2]
grid_width = width // grid_size
grid_height = heigth // grid_size

grids = []
for i in range(grid_size) : 
    for j in range(grid_size) : 
        x_min = j * grid_width
        y_min = i * grid_height
        x_max = (j+1) * grid_width
        y_max = (i+1) * grid_height
        grid = image_tensor[:,y_min:y_max, x_min:x_max]
        grids.append(grid)

fig, axs = plt.subplots(grid_size, grid_size, figsize=(10,10))

for i in range(grid_size) : 
    for j in range(grid_size) : 
        axs[i,j].imshow(grids[i*grid_size+j].permute(1,2,0))
        axs[i,j].axis('off')

plt.show()