import torch.nn as nn
latent_dim = 20 # 잠재 변수
class DenoisingAutoEncoder(nn.Module) :

    def __init__(self):
        super(DenoisingAutoEncoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            # input image size 32 x 32 -> 784
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded