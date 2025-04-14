import json
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as F

# 라벨 정보
labels = ['crease', 'crescent_gap', 'inclusion', 'oil_spot',
          'punching_hole', 'rolled_plt', 'silk_spot',
          'waist_folding', 'water_spot', 'welding_line']

def crop_and_save_image(json_path, output_dir, train_ratio=0.9) :
    with open(json_path ,'r',  encoding='utf-8') as f :
        json_data = json.load(f)

    # train, val folder create
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    """
    output_dir = "ex02_dataset"
    ex02_dataset 
        train
        val
    """
    # labels folder create
    for label in labels :
        train_label_dir = os.path.join(train_dir, label)
        os.makedirs(train_label_dir, exist_ok=True)
        val_label_dir = os.path.join(val_dir, label)
        os.makedirs(val_label_dir, exist_ok=True)

    for filename in tqdm(json_data.keys()) :
        json_image = json_data[filename]
        width = json_image['width']
        height = json_image['height']
        file_name = json_image['filename']
        bboxes = json_image['anno']

        # image loader
        image_path = os.path.join("./ex_02_data/images/", file_name)
        image = Image.open(image_path)
        image = image.convert("RGB")

        for bbox_idx, bbox in enumerate(bboxes) :
            label_name = bbox['label']
            bbox_xyxy = bbox['bbox']
            x1, y1, x2, y2 = bbox_xyxy

            # bounding box crop
            cropped_image = image.crop((x1, y1, x2, y2))

            # padding
            width_, height_ = cropped_image.size
            if width_ > height_ :
                padded_image = Image.new(cropped_image.mode, (width_,width_), (0,))
                padding = (0, int((width_ - height_) /2))
            else :
                padded_image = Image.new(cropped_image.mode, (height_, height_), (0,))
                padding = (int((height_ - width_)/2) ,0)

            padded_image.paste(cropped_image, padding)

            # image resize
            size=(255,255)
            resize_image = F.resize(cropped_image, size)

            # train val label folder image save
            if np.random.rand() < train_ratio :
                save_dir = os.path.join(train_dir, label_name)
            else :
                save_dir = os.path.join(val_dir, label_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{filename}_{label_name}_{bbox_idx}.png")
            padded_image.save(save_path)




if __name__ == "__main__" :
    json_path = "./ex_02_data/anno/annotation.json"
    output_dir = "./ex02_dataset"

    crop_and_save_image(json_path,output_dir)