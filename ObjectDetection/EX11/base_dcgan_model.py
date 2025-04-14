import torch.nn as nn
from base_dcgan_config import nc, nz, ndf, ngf

# images input size 64 x 64
class BaseDcganGenerator(nn.Module):
    def __init__(self):
        super(BaseDcganGenerator, self).__init__()
        self.main = nn.Sequential(
            # ConvTranspose
            # nz = 100, ngf = 64
            nn.ConvTranspose2d(nz, ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            # BatchNorm
            nn.BatchNorm2d(ngf*8),
            # ReLU
            nn.ReLU(),
            # True : 결과를 저장하기 위해 새 변수를 생성하지 않고 입력값을 직접 수정하여 활성화 함수를 적용합니다.
            # -> 메모리 효율성, 계산 효율성
            # (ngf x 8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            # (nfg * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),
            # (ngf * 2) x 16 x16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            # Tanh
            nn.Tanh()
            # (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)


class BaseDcganDiscriminator(nn.Module) :
    def __init__(self):
        super(BaseDcganDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            # 0.1 ~ 0.3
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(ndf*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

