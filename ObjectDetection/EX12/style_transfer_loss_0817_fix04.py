import torch.nn as nn
import torch
import torch.nn.functional as F

# ContentLoss
class ContentLoss(nn.Module) :

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# gram_matrix
def gram_matrix(input) :
    a, b, c, d = input.size() # (batch_size, map_number, height, width)

    features = input.view(a * b , c * d) # 입력 텐서  2d 텐서로 변형 -> 각각의 특성 맵을 백터 -> 하나의 행렬 합침

    G = torch.mm(features, features.t()) # 2D 특성 맵을 이용 해서 그램 행렬 계산 -> 특성 맵 간의 내적 계산 -> 유사도 측정 하는 역할

    return G.div(a * b * c * d) # 계산된 그램 형렬을 특성 맵의 전체 차원으로 나누어 -> 정규화 -> 행렬 크기에 따른 스케일 조정하여 일반화 역활

# StyleLoss
class StyleLoss(nn.Module) :
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)

        return input

