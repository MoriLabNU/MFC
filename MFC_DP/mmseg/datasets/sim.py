# Obtained from https://github.com/lhoyer/DAFormer
# Modifications: Support new dataset

from . import CholecSegDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SimDataset(CustomDataset):
    CLASSES = CholecSegDataset.CLASSES
    PALETTE = CholecSegDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(SimDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
