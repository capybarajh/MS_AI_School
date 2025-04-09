import cv2

# 동영상 파일 열기
cap = cv2.VideoCapture("./data/vtest.avi")

# ORB 객체 생성
orb = cv2.ORB_create()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 특징점 검출
    keypoints = orb.detect(gray, None)

    # 특징점 그리기
    frame = cv2.drawKeypoints(frame, keypoints, None, (0, 150, 220), flags=0)

    # 출력
    cv2.imshow("ORB", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
