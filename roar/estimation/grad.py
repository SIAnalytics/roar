import torch

from roar.registry import ATTRIBUTES
from .attribute import BaseAttribute


@ATTRIBUTES.register_module('Grad')
@ATTRIBUTES.register_module()
class Gradient(BaseAttribute):

    @torch.enable_grad()
    def _estimate(self, data_batch: dict) -> torch.Tensor:
        grad = torch.autograd.grad(
            self._run_forward(data_batch), data_batch['inputs'])[0]
        return grad

    @property
    def name(self) -> str:
        return 'grad'


@ATTRIBUTES.register_module('GI')
@ATTRIBUTES.register_module()
class GradientTimesInput(Gradient):

    def _estimate(self, data_batch: dict) -> torch.Tensor:
        return super(GradientTimesInput,
                     self)._estimate(data_batch) * data_batch['inputs']

    @property
    def name(self) -> str:
        return 'gi'
