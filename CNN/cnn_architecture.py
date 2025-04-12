import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        # self.conv2를 거친 특징값 행렬이 32채널 7x7 행렬로 나오는 예시이므로,
        # 마지막 완전연결층의 입력값이 32 * 7 * 7 로 계산됨

    def forward(self, x):
        x = self.conv1(x)
        # 첫번째 합성곱층 통과
        x = self.relu(x)
        # 활성함수 적용
        x = self.pool(x)
        # 풀링. 풀링 다음에는 활성함수를 다시 먹이지 않음
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.fc(x)
        # 출력층이기 때문에 활성함수를 먹이지 않음
        return x