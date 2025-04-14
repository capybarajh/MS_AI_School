import cv2
import json
import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.builder import  DATASETS
from mmdet.apis import set_random_seed
from mmcv import Config
from mmdet.datasets.coco import CocoDataset
import os

# fix code : add
from mmdet.models import build_detector


# config setting == train config setting
config_file = "./configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_file)

@DATASETS.register_module(force=True)
class ParkingDataset(CocoDataset) :
    CLASSES = ('세단(승용차)', 'SUV', '승합차', '버스','학원차량(통학버스)',
               '트럭','택시','성인','어린이','오토바이','전동킥보드','자전거','유모차','쇼핑카트')

# Dataset path setting
cfg.dataset_type = 'ParkingDataset'
cfg.data_root = './Data/'

cfg.data.train.type = "ParkingDataset"
cfg.data.train.ann_file = "./Data/train.json"
cfg.data.train.img_prefix = "./Data/images/"

cfg.data.val.type = "ParkingDataset"
cfg.data.val.ann_file = "./Data/valild.json"
cfg.data.val.img_prefix = "./Data/images/"

cfg.data.test.type = "ParkingDataset"
cfg.data.test.ann_file = "./Data/test.json"
cfg.data.test.img_prefix = "./Data/images/"

cfg.model.roi_head.bbox_head.num_classes = 14
# pre-train model path
cfg.load_from = "./dynamic_rcnn_r50_fpn_1x-62a3f276.pth"

# train checkpoint save dir path
cfg.work_dir = "./work_dirs/0804"

cfg.lr_config.warmup = None
cfg.log_config.interval = 10

cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 6
cfg.checkpoint_config.interval = 6

cfg.runner.max_epochs = 10
cfg.seed = 0
cfg.data.samples_per_gpu = 6
cfg.data.workers_per_gpu = 2
cfg.gpu_ids = range(1)
set_random_seed(0 , deterministic=False)

checkpoint_file_path = "./epoch_15.pth"

import torch
device = torch.device("cpu")
model = init_detector(cfg, checkpoint_file_path, device=device)


with open("./Data/test.json", 'r', encoding='utf-8') as f :
    image_infos = json.loads(f.read())

# Confidence Score def => 0.5

if __name__ == "__main__" :

    score_threshold = 0.5
    for img_info in image_infos['images'] :
        file_name = img_info['file_name']
        image_path = os.path.join("./Data/images/" , file_name)

        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = inference_detector(model, img)

        for idx, result in enumerate(results) :
            if len(result) == 0 :
                continue
                y_max = int(result_filterd[i,3])

                result_filterd = result[np.where(result[:, 4] > score_threshold)]

                for i in range(len(result_filterd)):
                    x_min = int(result_filterd[i, 0])
                    y_min = int(result_filterd[i, 1])
                    x_max = int(result_filterd[i, 2])
                print(x_min, y_min, x_max, y_max, float(result_filterd[i,4]))

                # cv2
                cv2.rectangle(img, (x_min, y_min) , (x_max, y_max), (0,255,0), 2)

            cv2.imshow("test", img)
            cv2.waitKey(0)

