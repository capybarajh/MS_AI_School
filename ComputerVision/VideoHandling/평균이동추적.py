import cv2

# 평균 이동 추적을 위한 초기 사각형 설정
track_window = None  # 객체의 위치 정보 저장할 변수
roi_hist = None  # 히스토그램을 저장할 변수
trem_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 동영상 파일 열기
cap = cv2.VideoCapture("./data/slow_traffic_small.mp4")

# 첫 프레임에서 추적할 객체 선택
ret, frame = cap.read()
print(frame)

x, y, w, h = cv2.selectROI("selectROI", frame, False, False)
print("선택한 박스 좌표 >> ", x, y, w, h)

# 추적할 객체의 초기 히스토그램 계산

roi = frame[y : y + h, x : x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# # 추적할 객체의 초기 윈도우 설정
track_window = (x, y, w, h)

cv2.imshow("roi test", roi)
cv2.waitKey(0)

""" 
현재 작업 디렉토리 기준 :  /data/image 라고 가정 한다면 : 
image - boon 폴더 안에 이미지 경로 가져오고 싶은경우 
상대경로 : ./boon/000.png (현재 작업 디렉토리 기준으로 파일 디렉토리 위치 지정방식)

절대경로 : /data/image/boon/000.png (전체 경로)
"""

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break
    # 추적할 객체의 히스토그램 역투명 계산
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 평균 이동 알고리즘을 통해 객체 위치 추정
    _, track_window = cv2.meanShift(dst, track_window, trem_crit)

    # 추적 결과를 사각형으로 표시
    x, y, w, h = track_window
    print("추적 결과 좌표 ", x, y, w, h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 프레임 출력
    cv2.imshow("Mean Shift Tracking ..", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord("q"):
        exit()

# 자원 해제
cap.release()
cv2.destroyAllWindows()