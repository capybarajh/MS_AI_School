import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

kernal_size = 15
kernal_direction = np.zeros((kernal_size, kernal_size))
kernal_direction[int((kernal_size)//2), :] = np.ones(kernal_size)
kernal_direction /= kernal_size # 커널의 합이 1이 되도록
kernal_matrix = cv2.getRotationMatrix2D((kernal_size/2, kernal_size/2), 45, 1)
kernal = np.hstack((kernal_matrix[:, :2], [[0], [0]]))
        # kernal_matrix[:, :2] <- 회전 행렬에서 병진이동 벡터를 제외하고 회전 행렬 값만 가져옴
        # [[0],[0]] <- 병진이동 벡터 (이동 X)
kernal = cv2.warpAffine(kernal_direction, kernal, (kernal_size, kernal_size))

motion_blur_img = cv2.filter2D(image, -1, kernal)
plt.imshow(motion_blur_img)
plt.show()
