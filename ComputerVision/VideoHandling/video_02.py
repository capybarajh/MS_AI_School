import cv2

# 비디오 파일 읽기
cap = cv2.VideoCapture("./data/blooms-113004.mp4")

# 비디오 정보 가져오기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# 비디오 정보 출력
print(width, height, fps, frame_count)

# 동영상 파일 읽기 예시
if cap.isOpened():  # 캡처 객체 초기화 확인 !!
    print("캡처 객체 초기화 확인 !!")
    while True:
        ret, frame = cap.read()  # 다음 프레임 읽기
        # ret -> 프레임 읽기가 성공했는지를 나타내는 부울 값
        # frame -> 이미지 numpy 배열 형태 -> 픽셀 정보
        if not ret:  # 프레임 읽기 실패 시 루프 종료
            break
        else:  # 프레임 읽기 성공 시 루프 실행
            # 프레임 크기 조정 -> 영상 크기 수정
            frame = cv2.resize(frame, (640, 480))
            # print(frame.shape) # (480, 640, 3)
            cv2.imshow("video test", frame)  # 화면 표시
            # q 버튼을 누르면 종료
            if cv2.waitKey(25) & 0xFF == ord("q"):
                exit()
else:
    print("캡처 객체 초기화 실패 !!")

# 카메라 자원 반납
cap.release()
cv2.destroyAllWindows()
