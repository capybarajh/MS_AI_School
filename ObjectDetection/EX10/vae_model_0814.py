import torch.nn as nn
import torch
# 10 ~ 100
latent_dim = 20

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * latent_dim)

        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid() # 0 ~ 1
        )
    """
     잠재 변수를 리파라미터화(reparameterization)하는 과정
     mu : 잠재 변수 mean value 
     layer : VAE의 인코더에서 나온 로그 분산(log variance) 값
    """
    def reparameterize(self, mu, layer): # 재매개변수화
        std = torch.exp(0.5 * layer) #  로그 분산 값 layer를 이용하여 표준 편차(standard deviation) 값을 계산
        # 로그 분산이 사용되는 이유는 분산의 범위가 양수여야 하기 때문에
        epsilon = torch.randn_like(std)
        # 함수는 주어진 텐서와 동일한 shape을 가진 텐서를 생성하며, 그 값은 평균이 0이고 표준 편차가 1인 정규 분포에서 샘플링한 값
        return mu + epsilon * std

    def forward(self, x):
        x = x.view(-1, 784) # input 2d -> 1d 784 dim
        mu_logval = self.encoder(x) # x - > encoder -> output -> mu, log variance type : tensor val
        mu = mu_logval[:, :latent_dim] # front -> mu val
        logvar = mu_logval[:,latent_dim:] # end -> log variance
        z = self.reparameterize(mu, logvar) # mu, logvar -> 정규 분포에서 z 값을 샘플링하여 잠재 변수 획득
        x_recon = self.decoder(z) # 잠재 변수 z를 입력하여 재구성된 이미지 x_recon 획득

        return x_recon, mu, logvar # 재구성된 이미지, 잠재 변수의 평균, 로그 분산 값