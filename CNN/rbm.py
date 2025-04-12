import torch
from torch import nn
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader

class RBM(nn.Module):
    def __init__(self, visible_size, hidden_size):
        super(RBM, self).__init__()
        self.W = nn.Parameter(
                torch.randn(visible_size, hidden_size)
            )
        # 최초 가중치 (난수로 임의생성됨)

        self.v_bias = nn.Parameter(
                torch.randn(visible_size)
            )
        # 가시층의 편향치

        self.h_bias = nn.Parameter(
                torch.randn(hidden_size)
            )
        # 은닉층의 편향치

    def forward(self, x):
        hidden_prob = torch.sigmoid(
                torch.matmul(x, self.W) + self.h_bias
                # xW + b
            )
        # 은닉층의 확률값
        hidden_state = torch.bernoulli(
            hidden_prob
        )
        # 은닉층 확률값을 토대로 나온 함수값
        visible_prob = torch.sigmoid(
            torch.matmul(hidden_state,
                        torch.transpose(self.W, 0, 1)
                         ) + self.v_bias
            # hidden_state에 가중치의 전치행렬을 사용
            # (0, 1) -> 출력값 크기 지정
        )

        return visible_prob, hidden_state
    

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5,))
    ])
    # 텐서화 한 다음 Normalize 진행하는 transforms

    train_dataset = torchvision.datasets.MNIST(
                        root="./data", train=True, transform=transform, download=True
                    )
    # MNIST 데이터셋 다운로드 후 train_dataset으로 불러옴

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # Batch Size 64로 지정한 데이터로더

    visible_size = 28 * 28
    # MNIST 데이터셋 이미지의 크기

    hidden_size = 256
    # 은닉층의 크기, 명확한 값이 정해지지 않았다면 관습적으로 256으로 생성

    rbm = RBM(visible_size, hidden_size)
    # RBM 모델의 실제 생성

    criterion = nn.BCELoss()
    # 오차값에 대한 손실함수 정의
    optimizer = torch.optim.SGD(rbm.parameters(), lr=0.01)
    # 오차값을 통해서 가중치를 조절할 최적화 알고리즘

    num_epochs = 10
    for epoch in range(num_epochs):
        for images, _ in train_loader:
            inputs = images.view(-1, visible_size)
            # train_loader에서 얻은 2차원 텐서를 일렬로 펼침

            visible_prob, _ = rbm(inputs)
            # 순전파

            loss = criterion(visible_prob, inputs)
            # ANN 코드와 비교 -> output 자리에 visible_prob, 확률값이 들어감
            #                -> 비교 대상이 입력값이므로 labels 자리에 inputs가 들어감

            optimizer.zero_grad()
            # 업데이트할 가중치 초기화
            loss.backward()
            # 역전파 계산
            optimizer.step()
            # 역전파 반영

        print(f"Epoch {epoch + 1} / {num_epochs} : Loss {loss.item():.4f}")

        vutils.save_image(rbm.W.view(hidden_size, 1, 28, 28), 
                          f"weights_{epoch+1}.png", normalize=True)
        # 가중치 이미지 저장 -> 노이즈가 나올 것임

        inputs_display = inputs.view(-1, 1, 28, 28)
        # 펼쳐진 텐서를 28 * 28 평면으로 재구성
        outputs_display = visible_prob.view(-1, 1, 28, 28)
        # 얻은 확률평면을 28 * 28로 재구성
        comparison = torch.cat([inputs_display, outputs_display], dim=3)
        
        vutils.save_image(comparison, f"reconstruction_epoch_{epoch+1}.png", normalize=True)
        # 결과값을 이미지로 확인
