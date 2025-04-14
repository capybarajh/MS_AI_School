import mmcv
import torch

from ex_0807_cfg import cfg
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv.runner import get_dist_info, init_dist
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes

@DATASETS.register_module(force=True)
class WebsiteScreenshotsDataset(CocoDataset) :
    CLASSES = ('elements','button', 'field', 'heading', 'iframe', 'image', 'label',
                'link', 'text')

if __name__ == "__main__" :
    datasets = [build_dataset(cfg.data.train)]
    print(datasets[0])

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg = cfg.get('test_cfg'))
    # print(datasets[0].__dict__.keys())
    # print(datasets[0].data_infos)
    model.CLASSES = datasets[0].CLASSES
    print(model.CLASSES)

    train_detector(model, datasets, cfg, distributed=False, validate=True)