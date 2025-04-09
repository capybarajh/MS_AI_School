import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image02.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

x_diff = 50
y_diff = 100
h, w, c = image.shape
M = np.float32([
    [1, 0, x_diff],
    [0, 1, y_diff]
]) # x축으로 50, y 축으로 100 이동하는 병진이동행렬
shifted_img = cv2.warpAffine(image, M, (w, h))

M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)
rotated_img = cv2.warpAffine(image, M, (w, h))

M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, 0.5)
halfed_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_AREA) # 가장자리를 검은색으로 칠한, 원본 이미지 크기와 같은 축소 이미지
croped_img = halfed_img[h//2 - h//4 : h//2 + h//4, 
                        w//2 - w//4 : w//2 + w//4] # 가장자리를 잘라낸 이미지

resized_img = cv2.resize(image, (w//2, h//2), interpolation=cv2.INTER_AREA)
plt.imshow(image)
plt.show()
plt.imshow(shifted_img)
plt.show()
plt.imshow(rotated_img)
plt.show()
plt.imshow(resized_img)
plt.show()
plt.imshow(halfed_img)
plt.show()
plt.imshow(croped_img)
plt.show()

#### blurring ####
blur_img = cv2.GaussianBlur(image, (5, 5), 5)

plt.imshow(blur_img)
plt.show()
