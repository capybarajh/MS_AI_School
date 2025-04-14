import torch
import torch.nn as nn
import torch.optim as optim
# from base_dcgan_model import BaseDcganGenerator, BaseDcganDiscriminator
# from base_dcgan_train import weights_init

# base_dcgan_config
nz = 100 # 잠재공간 벡터의 크기
ngf = 64 # 생성자를 통과하는 특징 데이터들의 채널 크기
ndf = 64 # 판별자를 통과하는 특징 데이터들의 채널 크기
nc = 3 # RGB 이미지이기 때문에 3 으로 설정합니다.

# base_dcgan_train_config
data_root = "./cat_dataset/"
num_workers = 2
batch_size = 128
img_size = 64 # all image size 64 x 64
num_epochs = 200
lr = 0.00025
# Adam, AdamW -> bata1 = 0.2  ~ 0.4
beta1 = 0.4
device = 'cuda' if torch.cuda.is_available() else 'cpu'



