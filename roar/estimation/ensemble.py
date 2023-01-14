from abc import abstractmethod
from functools import partial

import torch
from mmengine.model import BaseModel

from roar.registry import ATTRIBUTES
from .attribute import BaseAttribute


class EnsembleGradients(BaseAttribute):
    REDUCTION = dict(mean=torch.mean, var=torch.var)
    reduction = 'mean'

    def __init__(self,
                 model: BaseModel,
                 iter: int,
                 attr_cfg: dict = dict(type='Grad'),
                 **kwargs):
        super(EnsembleGradients, self).__init__(model, **kwargs)
        self.iter = iter
        self.attr = ATTRIBUTES.build(attr_cfg, default_args=dict(model=model))

    @abstractmethod
    def _wrap_batch(self, i: int, data_batch: dict) -> dict:
        pass

    def _estimate(self, data_batch: dict) -> torch.Tensor:
        attr = self.REDUCTION[self.reduction](
            torch.stack([
                self.attr(self._wrap_batch(i, self._copy(data_batch)))
                for i in range(self.iter)
            ]),
            dim=0)
        return attr


@ATTRIBUTES.register_module('IG')
@ATTRIBUTES.register_module()
class IntegratedGradients(EnsembleGradients):
    BASELINE = dict(
        zero=torch.zeros_like,
        mean=partial(torch.mean, dim=(2, 3), keepdim=True))

    def __init__(self,
                 model: BaseModel,
                 iter: int = 25,
                 baseline: str = 'zero',
                 **kwargs):
        super(IntegratedGradients, self).__init__(model, iter=iter, **kwargs)
        self.iter = iter
        self.baseline = baseline

    def _wrap_batch(self, i: int, data_batch: dict) -> dict:
        alpha = i / self.iter
        baseline = data_batch.pop('baseline')
        data_batch['inputs'] = baseline + alpha * (
            data_batch['inputs'] - baseline)
        return data_batch

    @torch.enable_grad()
    def _estimate(self, data_batch: dict) -> torch.Tensor:
        baseline = self.BASELINE[self.baseline](data_batch['inputs'])

        data_batch['baseline'] = baseline
        return (data_batch['inputs'] - baseline) * super(
            IntegratedGradients, self)._estimate(data_batch)

    @property
    def name(self) -> str:
        if self.attr.name == 'grad':
            return 'ig'
        return f'ig-{self.attr.name}'


@ATTRIBUTES.register_module('SG')
@ATTRIBUTES.register_module()
class SmoothGrad(EnsembleGradients):
    reduction = 'mean'

    def __init__(self,
                 model: BaseModel,
                 iter: int = 15,
                 sigma: float = 0.15,
                 **kwargs):
        super(SmoothGrad, self).__init__(model, iter=iter, **kwargs)
        self.sigma = sigma

    def _wrap_batch(self, _: int, data_batch: dict) -> dict:
        data_batch['inputs'] = data_batch['inputs'] + torch.normal(
            0,
            torch.ones_like(data_batch['inputs']) * self.sigma)
        return data_batch

    @property
    def name(self) -> str:
        if self.attr.name == 'grad':
            return 'sg'
        return f'sg-{self.attr.name}'


@ATTRIBUTES.register_module('VG')
@ATTRIBUTES.register_module()
class VarGrad(SmoothGrad):
    reduction = 'var'

    @property
    def name(self) -> str:
        if self.attr.name == 'grad':
            return 'vg'
        return f'vg-{self.attr.name}'
