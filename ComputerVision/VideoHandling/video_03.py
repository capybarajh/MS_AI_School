import cv2
cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)
# 1280 x 720 -> 종횡비에 맞게 이미지 크기 수정 16:9 -> 854×480

while True:
    ret, frame = cap.read()
    if ret:
        # 화면 사이즈 조절
        frame = cv2.resize(frame, (854, 480))

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
