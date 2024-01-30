# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional datasets

from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cholecseg import CholecSegDataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .sim import SimDataset
from .uda_dataset import UDADataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CholecSegDataset',
    'SimDataset',
    'UDADataset',
]
