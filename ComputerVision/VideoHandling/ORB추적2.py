import cv2
import numpy as np

# 동영상 파일 읽기
cap = cv2.VideoCapture("./data/vtest.avi")

# ORB 객체 생성
orb = cv2.ORB_create()

# 특징점 최소 크기 설정
min_keypoint_size = 10

# 중복 특징점 저거 기준 거리
duplicate_threshold = 10

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("프레임 읽기 실패")
        break

    # 그레이스 스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 특징점 검출
    keypoints = orb.detect(gray, None)

    # 특징점 크기가 일정 크기 이상인 것만 남기기
    keypoints = [kp for kp in keypoints if kp.size > min_keypoint_size]

    # 중복된 특징점 제거
    mask = np.ones(len(keypoints), dtype=bool)
    for i, kp1 in enumerate(keypoints):
        if mask[i]:
            for j, kp2 in enumerate(keypoints[i + 1 :]):
                if (
                    mask[i + j + 1]
                    and np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt))
                    < duplicate_threshold
                ):
                    mask[i + j + 1] = False
    keypoints = [kp for i, kp in enumerate(keypoints) if mask[i]]

    # 특징점 그리기
    frame = cv2.drawKeypoints(frame, keypoints, None, (0, 200, 150), flags=0)

    cv2.imshow("ORB", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
