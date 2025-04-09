import numpy as np
import imgaug.augmenters as iaa # ImgAug Augmenters
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("sample_data_01\\train\\snow\\0830.jpg")

image_arr = np.array(image)
image_arr.shape

images = [image_arr, image_arr, image_arr, image_arr]

# Affine 변환
rotate = iaa.Affine(rotate=(-25, 25))
images_aug = rotate(images=images)

# plt.figure(figsize=(12,12))
# plt.imshow(np.hstack(images_aug))
# plt.show()

# crop 실습
crop = iaa.Crop(percent=(0, 0.2))
images_aug01 = crop(images=images)

# plt.figure(figsize=(12,12))
# plt.imshow(np.hstack(images_aug01))
# plt.show()

# Sequential을 이용한 여러가지 Augmentation 기법
# ex1
# rotate_crop = iaa.Sequential([
#     iaa.Affine(rotate=(-25, 25)),
#     iaa.Crop(percent=(0, 0.2))
# ], random_order=True)
# images_aug02 = rotate_crop(images=images)
# plt.figure(figsize=(12,12))
# plt.imshow(np.hstack(images_aug02))
# plt.show()

# ex2
# rotate_crop = iaa.Sequential([
#     iaa.Crop(percent=(0, 0.2)),
#     iaa.Affine(rotate=(-25, 25))
# ], random_order=True)
# images_aug03 = rotate_crop(images=images)
# plt.figure(figsize=(12,12))
# plt.imshow(np.hstack(images_aug03))
# plt.show()

# iaa.OneOf() 사용
# seq = iaa.OneOf([
#     iaa.Grayscale(alpha=(0.0, 1.0)),
#     iaa.AddToSaturation((-50, 50))
# ])
# images_aug04 = seq(images=images)

# plt.figure(figsize=(12,12))
# plt.imshow(np.hstack(images_aug04))
# plt.show()

# iaa.Sometimes() 사용
seq = iaa.Sequential([
    iaa.Sometimes(
        0.6,
        iaa.AddToSaturation((-50, 50))
    ),
    iaa.Sometimes(
        0.2,
        iaa.Grayscale(alpha=(0.0, 1.0))
    )
])
images_aug05 = seq(images=images)

plt.figure(figsize=(12,12))
plt.imshow(np.hstack(images_aug05))
plt.show()