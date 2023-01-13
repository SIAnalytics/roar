from functools import reduce
from typing import Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseModel, BaseModule

from roar.registry import ATTRIBUTES
from .attribute import BaseAttribute


def getattr_recursive(module, name):
    return reduce(getattr, name.split('.'), module)


@ATTRIBUTES.register_module('GC')
@ATTRIBUTES.register_module()
class GradCAM(BaseAttribute):

    def __init__(self, model: BaseModel, module: Union[BaseModule, str],
                 **kwargs):
        super(GradCAM, self).__init__(model, **kwargs)
        if isinstance(module, str):
            module = getattr_recursive(model, name=module)
        self.module = module

    @torch.enable_grad()
    def _estimate(self, data_batch: dict) -> torch.Tensor:
        try:
            self._register_hooks()
            aFax = torch.autograd.grad(  # noqa
                self._run_forward(data_batch), data_batch['inputs'])[0]
            attr = self.f * self.aFaf.mean(dim=(2, 3), keepdim=True)
        finally:
            self._unregister_hooks()
        return F.interpolate(
            attr, data_batch['inputs'].shape[-2:], mode='bilinear')

    def _register_hooks(self):
        self._handles = [
            self.module.register_forward_hook(self._forward_hook),
            self.module.register_backward_hook(self._backward_hook)
        ]

    def _unregister_hooks(self):
        for handle in self._handles:
            handle.remove()
        del self._handles

    def _forward_hook(self, module, input, output: Union[torch.Tensor, tuple]):
        self.f = output.clone().detach()

    def _backward_hook(self, module, grad_input,
                       grad_output: Union[torch.Tensor, tuple]):
        self.aFaf = grad_output[0].clone().detach()

    @property
    def name(self) -> str:
        return 'gc'
