from mmengine.hooks import Hook

from mmcls.registry import HOOKS


@HOOKS.register_module()
class FeatureEstimationHook(Hook):
    pass