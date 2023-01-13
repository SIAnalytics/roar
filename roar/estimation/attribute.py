import copy
from abc import ABC, abstractmethod
from typing import Union

import torch
from mmengine.model import BaseModel


class BaseAttribute(ABC):
    """Base class for estimating a feature importance.

    Args:
        model (BaseModel): The model used to estimate a feature importance
        label (int | str): The target logit. If label is a str, it must be
            either 'gt_label' or 'pred_label'. Defaults to 'pred_label'
        use_abs (bool): Whether to take the absolute value of a feature
            importance. Defaults to True
        use_pos (bool): Whether to apply relu to the feature importance.
            Defaults to False
    """

    def __init__(self,
                 model: BaseModel,
                 label: Union[int, str] = 'pred_label',
                 use_abs: bool = True,
                 use_pos: bool = False):
        self.model = model
        self.label = label
        self.use_abs = use_abs
        self.use_pos = use_pos

    def __call__(self, data_batch: dict) -> torch.Tensor:
        data_batch = self._copy(data_batch)
        attr = self._estimate(data_batch).detach()
        if self.use_pos:
            attr = torch.relu(attr)
        elif self.use_abs:
            attr = torch.abs(attr)
        return attr

    def _copy(self, data_batch: dict) -> dict:
        data_batch['inputs'] = data_batch['inputs'].detach()
        data_batch = copy.deepcopy(data_batch)
        data_batch['inputs'].requires_grad_()
        return data_batch

    def _run_forward(self, data_batch: dict) -> torch.Tensor:
        data = self.model.data_preprocessor(data_batch, False)
        logits = self.model(**data, mode='tensor')

        if isinstance(self.label, int):
            indices = [self.label] * len(data_batch['data_samples'])
        elif self.label == 'gt_label':
            indices = [
                data_sample.gt_label.get('label')
                for data_sample in data_batch['data_samples']
            ]
        elif self.label == 'pred_label':
            indices = logits.argmax(dim=1)
        else:
            raise ValueError

        return torch.sum(logits[:, indices])

    @abstractmethod
    def _estimate(self, data_batch: dict) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
