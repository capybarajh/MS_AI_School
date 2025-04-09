import cv2

# 동영상 파일 열기
cap = cv2.VideoCapture('C:\\Users\\ljh29\\OneDrive\\바탕 화면\\MS AI School\\이미지분석\\0612_실습\\data\\slow_traffic_small.mp4')

# SIFT 객체 생성
sift = cv2.SIFT_create()

# 첫 프레임 읽기
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 특징점 검출
prev_keypoints, prev_descriptors = sift.detectAndCompute(prev_gray, None)

# 추적할 객체 선택
bbox = cv2.selectROI('Select Object', prev_frame, False, False)

# 추적을 위한 초기 추정 위치 설정
x, y, w, h = bbox
track_window = (x, y, w, h)

# 추적기 초기화
roi = prev_gray[y:y+h, x:x+w]
roi_keypoints, roi_descriptors = sift.detectAndCompute(roi, None)
matcher = cv2.BFMatcher(cv2.NORM_L2)
matches = matcher.match(prev_descriptors, roi_descriptors)
matches = sorted(matches, key=lambda x: x.distance)
matching_indices = [m.trainIdx for m in matches]

# 추적 결과를 표시하기 위한 색상 설정
color = (0, 255, 0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
            break
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 추적하기
    roi_gray = gray[y:y+h, x:x+w]
    roi_keypoints, roi_descriptors = sift.detectAndCompute(roi_gray, None)
    matches = matcher.match(prev_descriptors, roi_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    # 추적 결과 그리기
    for match in matches:
        pt1 = prev_keypoints[match.queryIdx].pt
        pt2 = roi_keypoints[match.trainIdx].pt
        x1, y1 = map(int, pt1)
        x2, y2 = map(int, pt2)
        cv2.circle(frame, (x+x2, y+y2), 3, color, -1)

    # 프레임 출력
    cv2.imshow('Object Tracking', frame)

    # 'q'키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 이전 프레임과 추적 정보 업데이트
    prev_gray = gray.copy()
    prev_keypoints = roi_keypoints
    prev_descriptors = roi_descriptors

# 자원 해제
cap.release()
cv2.destroyAllWindows()