import cv2

# 동영상 파일 읽기
video_capture = cv2.VideoCapture("./example.mp4")

# 초기 프레임에서 추정할 객체의 영역 선택 (마우스로 드래그)
ret, frame = video_capture.read()

# print(ret,frame)

bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

print("bbox info >> ", bbox)

# 초기 추적 대상 설정
roi = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 평균이동 트래킹 설정
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) # 종료 기준 설정
roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0,180])
# [0] -> Hue 채널 / 180 히스토그램 bin 개수 설정 / 색상범위 0 ~ 179 까지 가짐
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

while True:
    ret, frame = video_capture.read()

    # 현재 프레임에서 대상 객체 추적
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # HSV 변환 - 색상, 채도, 명도 정보를 이용하기 위해서
    back_proj = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], 1)
    # 역투영을 수행 -> 객체의 히스토그램과 주어진 이미지의 픽셀 강도를 비교하여 객체 확률적인 위치 계산
    ret, bbox_ = cv2.meanShift(back_proj, bbox, term_criteria)
    # cv2.meanShift() -> 역투영 이미지, 초기 추적 대상 bbox 정보, 종료 기준
    print(bbox_)

    # 추적 결과 시각화
    x, y, w, h = bbox_
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("Tracking Test", frame)

    # 'q'키를 누르면 종료
    if cv2.waitKey(60) & 0xFF == ord('q'):
        exit()