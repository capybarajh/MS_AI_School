# pip install flask 

# 필요한 파일 : imagenet1000_labels_dict / vgg11.pt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from flask import Flask, jsonify, request

from imagenet1000_lables_temp import lable_dicr
from vgg11 import VGG11

app = Flask(__name__)

# 모델 로드 함수 
def load_model(model_path) : 
    model = VGG11(num_classes=1000)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

# 모델 로드 
model_path = "./vgg11-bbd30ac9.pth"
model = load_model(model_path)

# 이미지 전처리 함수 
def preprocess_image(image) : 
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
    ])
    image = transform(image).unsqueeze(0)

    return image

# API 엔드포인트 설정
@app.route('/predict', methods=['POST'])
def predict() : 
    if 'image' not in request.files : 
        return jsonify({'error ': 'No image uploaded' }), 400
    
    image = request.files['image']
    img = Image.open(image)
    img = preprocess_image(img)

    # 예측 
    with torch.no_grad() : 
        outputs = model(img)
        _, pred = torch.max(outputs.data, 1)

        label_number = int(pred.item())
        class_name = lable_dicr[label_number]

        prediction = str(class_name)

    return jsonify({'predictions':prediction}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)
    # app.run(debug=True)