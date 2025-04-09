import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 이동할 거리 설정(x,y)
shift = (0, 50)

# 변환 행렬 생성
M = np.float32([
    [1, 0, shift[0]],
    [0, 1, shift[1]]
])
# 이동 행렬: 좌측 2x2 -> 회전 행렬 (현재 단위행렬), 우측 1열: 이동 행렬 (x 변위, y 변위)

# 이동 변환 적용
shifted_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

plt.imshow(shifted_img)
plt.show()