import os
import matplotlib.pyplot as plt
import random

from pycocotools.coco import COCO
from PIL import Image

# label 1 - > damage
annfile_path = "./car_damage_dataset/train/COCO_train_annos.json"
mul_annfile_path = "./car_damage_dataset/train/COCO_mul_train_annos.json"
img_path = "./car_damage_dataset/img/"

# coco dataset API init
coco = COCO(annfile_path)
mul_coco = COCO(mul_annfile_path)
"""
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
"""
# class info
cats = coco.loadCats(coco.getCatIds())
coco_class_name = [cat['name'] for cat in cats]
print("COCO categories for damages \n{}\n".format(', '.join(coco_class_name)))

mul_cats = mul_coco.loadCats(mul_coco.getCatIds())
mul_class_name = [cat['name'] for cat in mul_cats]
print("COCO categories for damages \n{}\n".format(', '.join(mul_class_name)))

"""
COCO categories for damages 
damage

COCO categories for damages 
headlamp, rear_bumper, door, hood, front_bumper
"""

catIds = coco.getCatIds(catNms=['damage'])
imgIds = coco.getImgIds(catIds = catIds)
random_img_id = random.choice(imgIds)
# print(f"{random_img_id} image id was selected at random from the {imgIds} ")
# load the image
imgIds = coco.getImgIds(imgIds=[random_img_id])
print(imgIds)
img = coco.loadImgs(imgIds)[0]
# {'coco_url': '', 'date_captured': '2020-07-14 09:59:34.190485',
# 'file_name': '21.jpg', 'flickr_url': '', 'height': 1024, 'id': 10, 'license': 1, 'width': 1024}
image_path = os.path.join(img_path, img['file_name'])
print(image_path)
image = Image.open(image_path)
plt.axis('off')
plt.imshow(image)
plt.show()
