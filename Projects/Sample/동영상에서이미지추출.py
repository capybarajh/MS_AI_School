import cv2

cap = cv2.VideoCapture('./홈페이지 배경 샘플 영상 - 바다.mp4')

print('동영상 프레임 수 :', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('동영상 가로 길이 :', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('동영상 세로 길이 :', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('FPS :', int(cap.get(cv2.CAP_PROP_FPS)))

import os

# 저장할 이미지 개수 초기화
img_count = 0

# 이미지 저장하는 폴더 구성
os.makedirs("./video_frame_dataset/", exist_ok=True)

# 프레임 단위로 이미지 저장하기
while True:
    # 프레임 읽어오기
    ret, frame = cap.read()

    # 동영상 끝까지 읽으면 종료
    if not ret:
        break
    
    # 15프레임 단위로 이미지 저장
    if img_count % 15 == 0:
        # 이미지 저장하기
        img_filename = f'./video_frame_dataset/frame_{img_count:04d}.png'

        cv2.imwrite(img_filename, frame)

    img_count += 1

# 동영상 파일 닫기
cap.release()