import random
import cv2
import albumentations as A
import matplotlib.pyplot as plt

image = cv2.imread('cat_dog.jpeg')

def visualize(tf_img):
    cv2.imshow("org", img)
    cv2.imshow("tf_img", tf_img)
    cv2.waitKey()

#### 좌우반전
# transform = A.HorizontalFlip(p=0.5)
# random.seed(7)
# augmentated_img = transform(image=image)['image']
# cv2.imshow("", augmentated_img)
# cv2.waitKey()
#### 좌우반전


#### ShiftScaleRotate
# transform = A.ShiftScaleRotate(p=0.5)
# random.seed(7)
# augmentated_img = transform(image=image)['image']
# print(augmentated_img)
# cv2.imshow("org", image)
# cv2.imshow("", augmentated_img)
# cv2.waitKey()
#### ShiftScaleRotate

#### Compose
# transform = A.Compose([
#     A.CLAHE(),
#     A.Flip(),
#     # A.OneOf([
#     #     A.IAAAdditiveGaussianNoise(),
#     #     A.GaussNoise()
#     # ], p=0.2),
#     A.RandomRotate90(),
#     A.Transpose(),
#     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
#     A.Blur(blur_limit=3),
#     A.OpticalDistortion(),
#     A.GridDistortion(),
#     A.HueSaturationValue()
# ])
# while True:
#     augmentated_img = transform(image=image)["image"]
#     cv2.imshow("org", image)
#     cv2.imshow("tf", augmentated_img)
#     key = cv2.waitKey()
#     if key == ord('q'):
#         cv2.destroyAllWindows()
#         break
#### Compose

img = cv2.imread("image02.jpeg")


# ### RandomRain
# transform = A.Compose(
#     [A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)]
# )
# rain_img = transform(image=img)['image']
# cv2.imshow("org", img)
# cv2.imshow("rain", rain_img)
# cv2.waitKey()
# ### RandomRain


#### RandomSnow
# transform = A.Compose(
#     [A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1)]
# )
# snow_img = transform(image=img)['image']
# cv2.imshow("org", img)
# cv2.imshow("snow", snow_img)
# cv2.waitKey()
#### RandomSnow

#### RandomSunFlare
# transform = A.Compose(
#     [A.RandomSunFlare(flare_roi = (0, 0, 1, 0.5), angle_lower=0.5, p=1)]
# )
# flare_img = transform(image=img)['image']
# cv2.imshow("org", img)
# cv2.imshow("flare", flare_img)
# cv2.waitKey()
#### RandomSunFlare


### RandomShadow
# transform = A.Compose(
#     [A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1)]
# )
# shadow_img = transform(image=img)['image']
# visualize(shadow_img)
### RandomShadow

#### RandomFog
# transform = A.Compose(
#     [A.RandomFog(fog_coef_lower=0.4, fog_coef_upper=0.6, alpha_coef=0.1, p=1)]
# )
# fog_img = transform(image=img)['image']
# visualize(fog_img)
#### RandomFog