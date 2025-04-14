import json
import os
from shutil import copy

def convert_coco_to_yolo(coco_json_file, org_images_folder, images_folder, labels_folder):
    with open(coco_json_file, 'r') as f:
        data = json.load(f)
    images = data['images']
    annots = data['annotations']
    
    for image in images :
        org_file_name = image['file_name']
        file_name = image['file_name'].split('.jpg')[0]
        id = image['id']
        width, height = image['width'], image['height']
        for annot in annots :
            if annot['image_id'] == id :
                category_id = annot['category_id'] - 1
                x, y, w, h = annot['bbox']
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w /= width
                h /= height

                # image copy to dst folder
                image_org_path = os.path.join(org_images_folder, org_file_name)
                image_dst_path = os.path.join(images_folder, org_file_name)
                copy(image_org_path, image_dst_path)

                # write to text file
                yolo_ann = f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                txt_file_path = os.path.join(labels_folder, f"{file_name}.txt")
                with open(txt_file_path, 'a') as f:
                    f.write(yolo_ann)

train_path = "./ultralytics/cfg/glass_dataset/train/"
valid_path = "./ultralytics/cfg/glass_dataset/valid/"
test_path = "./ultralytics/cfg/glass_dataset/test/"

train_yolo_path = "./ultralytics/cfg/glass_yolo_dataset/train/"
valid_yolo_path = "./ultralytics/cfg/glass_yolo_dataset/valid/"
test_yolo_path = "./ultralytics/cfg/glass_yolo_dataset/test/"

train_coco_json_file_path = os.path.join(train_path, "_annotations.coco.json")
valid_coco_json_file_path = os.path.join(valid_path, "_annotations.coco.json")
test_coco_json_file_path = os.path.join(test_path, "_annotations.coco.json")

train_images_path = os.path.join(train_yolo_path, "images")
train_labels_path = os.path.join(train_yolo_path, "labels")

valid_images_path = os.path.join(valid_yolo_path, "images")
valid_labels_path = os.path.join(valid_yolo_path, "labels")

test_images_path = os.path.join(test_yolo_path, "images")
test_labels_path = os.path.join(test_yolo_path, "labels")

os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)

os.makedirs(valid_images_path, exist_ok=True)
os.makedirs(valid_labels_path, exist_ok=True)

os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

convert_coco_to_yolo(train_coco_json_file_path, train_path, train_images_path, train_labels_path)
convert_coco_to_yolo(valid_coco_json_file_path, valid_path, valid_images_path, valid_labels_path)
convert_coco_to_yolo(test_coco_json_file_path, test_path, test_images_path, test_labels_path)