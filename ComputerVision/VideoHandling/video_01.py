import cv2

# 비디오 파일 읽기
cap = cv2.VideoCapture("./data/blooms-113004.mp4")

# 비디오 정보 가져오기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
frmae_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# 비디오 정보 보기
print(f"Oringinal width and height : {width}x{height}")
print(f"fps : {fps}")
print(f"fram count : {frmae_count}")

"""
결과
Oringinal width and height : 1920.0x1080.0
fps : 29.97002997002997
fram count : 751.0
"""
