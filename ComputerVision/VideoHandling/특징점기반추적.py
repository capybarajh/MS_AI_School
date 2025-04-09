import cv2

# 동영상 파일 읽기
cap = cv2.VideoCapture("./data/slow_traffic_small.mp4")

# 코너 검출기 파라미터 설정
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# 출력
# {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7}

# 광학 흐름 파라미터 설정
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 첫 프레임 읽기
ret, prve_frame = cap.read()
prev_gray = cv2.cvtColor(prve_frame, cv2.COLOR_BGR2GRAY)
print(prev_gray.shape)

# 초기 추적 지점 선택
prev_corner = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
prev_points = prev_corner.squeeze()

# 추적 결과 표시하기 위한 색상 설정
color = (0, 255, 0)

while True:
    # 다음 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    # 현재 프레임 변환 -> 그레이 스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # 광학 흐름 계산
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_points, None, **lk_params
    )

    # 추적 결과를 표시
    for i, (prev_point, next_point) in enumerate(zip(prev_points, next_points)):
        x1, y1 = prev_point.astype(int)
        x2, y2 = next_point.astype(int)

        print(x1, y1, x2, y2)  # 169 189 169 189
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (x2, y2), 3, color, 3)

    # 프레임 출력
    cv2.imshow("...", frame)

    # 다음 프레임을 위해 변수 업데이트
    prev_gray = gray.copy()
    prev_points = next_points

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
