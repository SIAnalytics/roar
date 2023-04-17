import os.path as osp
from typing import Sequence, Union

import numpy as np
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmpretrain.registry import HOOKS
from mmpretrain.structures import DataSample
from roar.estimation import BaseAttribute
from roar.registry import ATTRIBUTES


@HOOKS.register_module()
class FeatureEstimationHook(Hook):

    def __init__(self, estimator: Union[dict, list], out_dir: str = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()

        if isinstance(estimator, dict):
            estimator = [estimator]
        self._estimators = estimator

        self.out_dir = out_dir

    def before_test(self, runner: Runner):
        # set out_dir
        if self.out_dir is None:
            self.out_dir = runner.work_dir

        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        # register estimators
        for i, cfg in enumerate(self._estimators):
            self._estimators[i] = ATTRIBUTES.build(
                cfg, default_args=dict(model=model))

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DataSample]):
        for estimator in self._estimators:
            self._estimate_features(estimator, data_batch)

    def _estimate_features(self, estimator: BaseAttribute, data_batch: dict):
        attr = estimator(data_batch)
        for i, data_sample in enumerate(data_batch['data_samples']):
            name = osp.basename(data_sample.get('img_path'))
            out_file = osp.join(self.out_dir, estimator.name, name)
            self._visualizer.visualize_cls(
                self._visualizer.draw_featmap(attr[i]),
                data_sample,
                draw_gt=False,
                draw_pred=False,
                draw_score=False,
                out_file=out_file,
                name=f'{estimator.name}/{name}')
            _, ext = osp.splitext(name)
            np.save(out_file.replace(ext, '.npy'), attr[i].cpu().numpy())
