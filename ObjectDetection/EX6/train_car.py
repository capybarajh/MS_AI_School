import mmcv
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets import build_dataset
from mmcv import Config
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import set_random_seed
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes

@DATASETS.register_module(force=True)
class ParkingDataset(CocoDataset) :
    CLASSES = ('세단(승용차)', 'SUV', '승합차', '버스','학원차량(통학버스)',
               '트럭','택시','성인','어린이','오토바이','전동킥보드','자전거','유모차','쇼핑카트')

# config file setting
# Dynamic RCNN model load
config_file = "./configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_file)

# Learning rate setting
# sing GPU = 0.0025
cfg.optimizer.lr = 0.0025

# dataset setting
cfg.dataset_type = "ParkingDataset"
cfg.data_root = "./Data/"

# train val tesst dataset type data_root ann_file img_prefix setting
cfg.data.train.type = "ParkingDataset"
cfg.data.train.ann_file = "./Data/train.json"
cfg.data.train.img_prefix = "./Data/images/"

cfg.data.val.type = "ParkingDataset"
cfg.data.val.ann_file = "./Data/valild.json"
cfg.data.val.img_prefix = "./Data/images/"

cfg.data.test.type = "ParkingDataset"
cfg.data.test.ann_file = "./Data/test.json"
cfg.data.test.img_prefix = "./Data/images/"

# class number setting
cfg.model.roi_head.bbox_head.num_classes = 14

# small size object -> change anchor = size 8 -> 4
cfg.model.rpn_head.anchor_generator.scales = [4]

# pretrained model load
cfg.load_from = './dynamic_rcnn_r50_fpn_1x-62a3f276.pth'

cfg.work_dir = "./work_dirs/0804"

cfg.lr_config.warmup = None
cfg.log_config.interval = 10

cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 6
cfg.checkpoint_config.interval = 6

# Epochs setting
# 8 * 12 -> 96
cfg.seed = 0
cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 2
cfg.gpu_ids = range(1)
cfg.device = "cuda"
set_random_seed(0, deterministic=False)

print(cfg.pretty_text)
datasets = [build_dataset(cfg.data.train)]
print(datasets[0])

datasets[0].__dict__.keys()

model =build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)

if __name__ == '__main__':
    train_detector(model, datasets, cfg, distributed=False, validate=True)