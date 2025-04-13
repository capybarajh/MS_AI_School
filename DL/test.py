import requests

# app.py <- 이미지 파일 경로 
IMAGE_PATH = "./cat.png"
# API 엔드포인트 URL 
CLASSIFICATION_MODEL_API_URL = 'http://127.0.0.1:80/predict'
with open(IMAGE_PATH, 'rb')  as f : 
    files = {'image': f}
    requests = requests.post(CLASSIFICATION_MODEL_API_URL, files=files)
# 응답 확인 체크 
if requests.status_code == 200 : 
    try : 
        prediction = requests.json()['predictions']
        print("예측 결과 >>> " , prediction)
    except Exception as e : 
        print("API 오류 ", str(e))
else : 
    print("API 오류 !! ", requests.text)