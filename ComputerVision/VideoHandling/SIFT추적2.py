import cv2

cap = cv2.VideoCapture("./data/vtest.avi")

# SIFT 객체 생성
sift = cv2.SIFT_create(contrastThreshold=0.02)

# 특징점 개수 제한 설정
max_keypoints = 100  # 원하는 최대 특징점 개수

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이 스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 특징점 검출
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    print(keypoints, descriptors)

    # 특정점 제한
    if len(keypoints) > max_keypoints:
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_keypoints]

    # 특징점 그리기
    frame = cv2.drawKeypoints(
        frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # 프레임 출력
    cv2.imshow("SIFT", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
