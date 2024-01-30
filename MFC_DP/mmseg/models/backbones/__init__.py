# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional backbones

from .mix_transformer import MixVisionTransformer, mit_b5
from .resnet import ResNetV1c

__all__ = [
    'ResNetV1c',
    'MixVisionTransformer',
    'mit_b5',
]
