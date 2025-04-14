import os
import cv2
import glob
import numpy as np
from mmdet.models import build_detector

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.apis import init_detector, inference_detector

@DATASETS.register_module(force=True)
class CarDamageDataset(CocoDataset) :
    CLASSES = ('damage',)

# config
config_file = "./configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_file)

# dataset
cfg.dataset_type = "CarDamageDataset"
cfg.data_root = "./car_damage_dataset/"

cfg.data.train.type = "CarDamageDataset"
cfg.data.train.ann_file = "./car_damage_dataset/train/COCO_train_annos.json"
cfg.data.train.img_prefix = "./car_damage_dataset/train/"

cfg.data.train.pipeline[2].img_scale=(500,500)
cfg.data.train.pipeline[3].flip_ratio=0.3

cfg.data.val.type = "CarDamageDataset"
cfg.data.val.ann_file = "./car_damage_dataset/val/COCO_val_annos.json"
cfg.data.val.img_prefix = "./car_damage_dataset/val/"

cfg.data.val.pipeline[1].img_scale=(500,500)

cfg.data.test.type = "CarDamageDataset"
cfg.data.test.ann_file = "./car_damage_dataset/val/COCO_val_annos.json"
cfg.data.test.img_prefix = "./car_damage_dataset/val/"

cfg.data.test.pipeline[1].img_scale=(500,500)

print(cfg.pretty_text)

# class number
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

cfg.model.rpn_head.anchor_generator.scales = [4]

cfg.device = "cuda"
set_random_seed(777, deterministic=False)

if __name__ == "__main__" :
    image_path = "./car_damage_dataset/test"
    image_path_list = glob.glob(os.path.join(image_path, "*.jpg"))
    checkpoint_file_path = "./work_dirs/0810_mask_rcnn_r50_fpn_1x/epoch_72.pth"

    # Build the model
    model = build_detector(cfg.model, train_cfg=cfg.get('train.cfg'), test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    model = init_detector(cfg, checkpoint_file_path, device=cfg.device)
    model.eval()

    for path in image_path_list :
        img = cv2.imread(path)

        # inference
        result = inference_detector(model, img)
        bbox_result, segm_result = result

        overlay = img.copy()

        for bbox, segm in zip(bbox_result[0], segm_result[0]) :
            x1, y1, x2, y2, score = bbox
            x1, y1, x2, y2 = map(int, [x1, y1, x2 ,y2]) # 정수 변환

            if score > 0.35 :
                mask = segm

                # 이진 마스크(Binary Mask)를 생성
                binary_mask = (mask > 0).astype(np.uint8) * 255
                # True인 값은 255로, False인 값은 0으로 유지됩니다.
                print(binary_mask)
                # 이진 마스크는 일반적으로 물체의 윤곽을 표시하거나 특정 영역을 표시하는 데 사용될 수 있습니다.

                overlay[binary_mask > 0, 0] = 0 # Blue channel
                overlay[binary_mask > 0, 1] = 255 # Green channel
                overlay[binary_mask > 0, 2] = 0    # Red channel

                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0),2)

        cv2.imshow("test", overlay)
        cv2.waitKey(0)