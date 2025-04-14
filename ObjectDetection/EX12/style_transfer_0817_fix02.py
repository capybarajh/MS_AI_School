import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from PIL import Image
from style_transfer_loss import ContentLoss, StyleLoss, gram_matrix


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

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers=content_layers_default,
                               style_layers=style_layers_default)  :
    """
    :param cnn:  컨볼루션 신경망 vgg model
    :param normalization_mean: 정규화(normalization)를 위한 평균 값
    :param normalization_std: 정규화를 위한 표준 편차 값
    :param style_img: 스타일 이미지
    :param content_img: 내용 이미지
    :param content_layers: 내용 비교에 사용될 레이어
    :param style_layers: 스타일 비교에 사용될 레이어
    """

    # model normalization : 평균과 표준 편차 값을 사용하여 정규화를 적용
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # content / style loss -> list
    content_losses = []
    style_losses = []

    # normalization 레이어만 포함하는 nn.Sequential로 초기화됩니다. 이것은 신경망의 시작점
    model = nn.Sequential(normalization)

    # 스타일 전이(style transfer)를 위한 모델 생성 중간 단계를 설명
    # CNN 모델의 각 레이어를 확인하고, 레이어의 유형에 따라 레이어를 추가
    i = 0 # conv -> count
    for layer in cnn.children() : # cnn.children() -> CNN 모델의 각 레이어를 순회
        if isinstance(layer, nn.Conv2d) :
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU) :
            name = f'relu_{i}'
            layer = nn.ReLU() # 레이어를 생성하여 layer를 덮어씌웁니다.
        elif isinstance(layer, nn.MaxPool2d) :
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else :
            raise RuntimeError('Unrecogized layer : {} '.format(layer.__class__.__name__))

        model.add_module(name, layer) # name에 설정된 이름과 해당 레이어를 모델에 추가

        if name in content_layers :  # content_layers 리스트에 포함되는 경우, 내용 손실(content loss)을 계산하고 모델에 추가
            # content loss add
            target = model(content_img).detach()
            content_loss = ContentLoss(target) # target을 대상으로 내용 손실을 생성
            model.add_module('content_loss_{}'.format(i), content_loss) # 내용 손실 객체를 모델에 추가
            content_losses.append(content_loss)
            # content_layers 리스트에 있는 각 레이어에서 내용 손실을 계산하고, 해당 손실을 모델에 추가

        if name in style_layers :
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)

    # 손실 레이어 이후의 레이어들은 주로 스타일 전이(style transfer)에서 손실을 계산하는 역할을 수행하며,
    # 이전 레이어들은 이미지에 대한 변형을 담당하거나 중간 특성을 추출하는 역할
    for i in range(len(model) -1, -1, -1) :
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) :
            break

    model = model[:(i + 1)]
    """
    현재까지 생성한 모델에서 i + 1개의 레이어만 남기고 나머지 레이어를 삭제하는 효과를 갖습니다. 
    이는 모델 구성 중간에 불필요한 레이어를 제거하거나 원하는 구조를 형성하기 위해 사용될 수 있습니다.
    """
    return model, style_losses, content_losses

def get_input_optimizer(input_img) :
    # 제한된 메모리를 사용하여 대규모의 비선형 최적화 문제를 효과적으로 해결
    optimizer = optim.LBFGS([input_img])

    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=500,
                       style_weight = 1000000, content_weight=1) :
    print("Building the style transfer model..")
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                     style_img, content_img)
    input_img.requires_grad_(True) # input_img는 스타일 트랜스퍼의 입력 이미지로서, 이 이미지에 대한 기울기를 계산 -> 역전파 계산
    model.requires_grad_(False) #  VGG 모델 에서 학습 중에는 그 가중치를 업데이트하지 않아야 합니다.

    optimizer = get_input_optimizer(input_img) # 스타일 트랜스퍼의 목표는 입력 이미지를 조금씩 수정하여 스타일과 컨텐츠를 잘 반영하는 이미지를 생성하는 것입니다.

    print('Optimizing .. ')
    run = [0]
    while run[0] <= num_steps :

        def closure(): #  내부에서 실행되는 클로저 함수
            with torch.no_grad() :
                input_img.clamp_(0,1) # 0과 1 사이로 클램핑하는 부분입니다. 이는 이미지의 픽셀 값이 올바른 범위 내에 있도록 보장

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            # style_score, content_score: 스타일 손실과 컨텐츠 손실을 저장할 변수
            for sl in style_losses :
                style_score += sl.loss
            for cl in content_losses :
                content_score += cl.loss

            # 스타일 손실과 컨텐츠 손실에 각각 가중치를 곱합니다.
            style_score *= style_weight
            content_score *= content_weight

            # 총 손실을 계산
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0 :
                print('run {} : '.format(run))
                print('Style loss : {:.4f} Content loss {:.4f}'.format(
                    style_score.item(), content_score.item()
                ))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0,1)

    return input_img

