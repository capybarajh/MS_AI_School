import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config

# config
config_file = "./configs/detr/detr_r50_8x2_150e_coco.py"
cfg = Config.fromfile(config_file)

# setting lr number
cfg.optimizer.lr = 0.0025

# Dataset setting
cfg.dataset_type = "WebsiteScreenshotsDataset"
cfg.data_root = "./Website_Screenshots_data/"

# train val test setting
cfg.data.train.type = "WebsiteScreenshotsDataset"
cfg.data.train.ann_file = "./Website_Screenshots_data/train/_annotations.coco.json"
cfg.data.train.img_prefix = "./Website_Screenshots_data/train/"

cfg.data.val.type = "WebsiteScreenshotsDataset"
cfg.data.val.ann_file = "./Website_Screenshots_data/valid/_annotations.coco.json"
cfg.data.val.img_prefix = "./Website_Screenshots_data/valid/"

cfg.data.test.type = "WebsiteScreenshotsDataset"
cfg.data.test.ann_file = "./Website_Screenshots_data/test/_annotations.coco.json"
cfg.data.test.img_prefix = "./Website_Screenshots_data/test/"

cfg.model.bbox_head.num_classes = 9

# pretrained model load
cfg.load_from = "./detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth"
# print(cfg.pretty_text)

cfg.work_dir = './work_dirs/0807/'

cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# CocoDataset -> metric bbox (bbox -> mAP iou threshold 0.5 ~ 0.95)
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 5

# model save
cfg.checkpoint_config.interval = 5

cfg.runner.max_epochs = 150
cfg.seed = 0
cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 2
cfg.gpu_ids = range(1)
cfg.device = "cuda"


