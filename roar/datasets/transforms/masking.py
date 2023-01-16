import os.path as osp
from typing import Optional

import numpy as np
import scipy.ndimage as ndimage
from mmcv.transforms import BaseTransform

from mmcls.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RemOve(BaseTransform):
    """
    **Required Keys:**
    - img
    - img_path
    **Modified Keys:**
    - img
    - mask_path
    - mask

    Args:
        mask_dir (str):
        attr (str):
        mean (list, optional):
        ratio (int, optional):
        filter (str, optional): 'none', 'maximum', or 'gaussian'
        maximum_kwargs (dict): kwargs for maximum filter.
        gaussian_kwargs (dict): kwargs for gaussian filter.
    """
    attr = None
    ratio = None
    filter = None

    def __init__(
            self,
            mask_dir: str,
            attr: Optional[str] = None,
            mean: Optional[list] = None,
            ratio: Optional[int] = None,
            filter: Optional[str] = None,  # max or gaussian
            maximum_kwargs: dict = dict(size=3),
            gaussian_kwargs: dict = dict(sigma=1.0),
    ):
        self.mask_dir = mask_dir
        self.mean = np.array(mean or [0] * 3)
        if attr is not None:
            self.attr = attr
        if ratio is not None:
            self.ratio = ratio
        if filter is not None:
            self.filter = filter
        self.maximum_kwargs = maximum_kwargs
        self.gaussian_kwargs = gaussian_kwargs

    def _apply_filter(self, mask):
        assert self.filter in ('none', 'maximum', 'gaussian')

        if self.filter == 'maximum':
            mask = ndimage.maximum_filter(mask, **self.maximum_kwargs)
        elif self.filter == 'gaussian':
            mask = ndimage.gaussian_filter(mask, **self.gaussian_kwargs)
        return mask

    def transform(self, results: dict) -> dict:
        mask_path = osp.join(
            self.mask_dir, self.attr,
            osp.splitext(osp.basename(results['img_path']))[0] + '.npy')
        mask = np.expand_dims(np.load(mask_path).mean(axis=0), axis=-1)
        mask = self._apply_filter(mask)
        mask = mask >= np.percentile(mask, 100 - self.ratio)

        results['img'] = results['img'] * (1 - mask) + mask * self.mean
        results['mask'] = mask
        results['mask_path'] = mask_path

        return results

    def __repr__(self):
        pass
