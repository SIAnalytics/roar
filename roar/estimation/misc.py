import numpy as np
import scipy.ndimage as ndimage
import torch

from roar.registry import ATTRIBUTES
from .attribute import BaseAttribute


@ATTRIBUTES.register_module('Sobl')
@ATTRIBUTES.register_module()
class Sobel(BaseAttribute):

    def _sobel(self, img: torch.Tensor) -> torch.Tensor:
        img = img.detach().cpu().numpy()
        img = img.mean(axis=0)
        return torch.tensor(np.expand_dims(ndimage.sobel(img), axis=0))

    def _estimate(self, data_batch: dict) -> torch.Tensor:
        return torch.stack([self._sobel(img) for img in data_batch['inputs']])

    @property
    def name(self) -> str:
        return 'sobl'


@ATTRIBUTES.register_module('Rand')
@ATTRIBUTES.register_module()
class Random(BaseAttribute):

    def _estimate(self, data_batch: dict) -> torch.Tensor:
        return torch.rand_like(data_batch['inputs'])

    @property
    def name(self) -> str:
        return 'rand'
