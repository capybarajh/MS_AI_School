##### 25FPS 기준으로 프레임 나눠서 저장
import cv2
import os

# 비디오 파일 읽기
cap = cv2.VideoCapture("./data/blooms-113004.mp4")

# FPS 지정
fps = 25

count = 0
if cap.isOpened():
    while True:
        ret, frame = cap.read()

        if ret:
            if int(cap.get(1)) % fps == 0:
                # fps 25
                os.makedirs("./frame_image_data/", exist_ok=True)
                cv2.imwrite(
                    f"./frame_image_data/image_{str(count).zfill(4)}.png", frame
                )

                count = count + 1

        else:
            break

cap.release()
cv2.destroyAllWindows()
