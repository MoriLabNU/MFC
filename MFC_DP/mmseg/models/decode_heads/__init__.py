# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .daformer_head import DAFormerHead
from .dlv2_head import DLV2Head
from .hrda_head import HRDAHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead

__all__ = [
    'DepthwiseSeparableASPPHead',
    'DLV2Head',
    'SegFormerHead',
    'DAFormerHead',
    'HRDAHead',
]
