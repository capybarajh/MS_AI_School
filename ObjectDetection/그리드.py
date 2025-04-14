# 그리드가 어떻게 객체 탐지 모델의 입력 데이터 구성하는지 시각화 
import cv2
import numpy as np

# 이미지 불러오기 
image = cv2.imread("./image01.jpg")
print(image)

# 그리드 셀 크기 설정 
grid_size = (50,50) # width, height

def create_grid(image, grid_size=grid_size):
    height, width =image.shape[:2]
    grid_width, grid_height = grid_size

    # 그리드 생성
    grid_image = np.copy(image)
    for x in range(0, width, grid_width) :
        cv2.line(grid_image, (x,0),(x,height), (255,255,255), 1)
    for y in range(0, height, grid_height) : 
        cv2.line(grid_image, (0,y),(width, y), (255,255,255), 1)

    return grid_image

# 이미지를 그리드로 분할한 결과 확인
grid_image = create_grid(image, grid_size)

cv2.imshow("org" , image)
cv2.imshow("grid", grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    