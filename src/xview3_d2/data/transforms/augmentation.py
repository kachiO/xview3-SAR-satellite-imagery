import inspect
import logging
from typing import Tuple

import cv2
import numpy as np
from detectron2.config import CfgNode
from detectron2.data.transforms import Augmentation, augmentation_impl
from detectron2.utils.registry import Registry
from fvcore.transforms.transform import Transform

logger = logging.getLogger("detectron2.Augmentations")

AUGMENTATION_REGISTRY = Registry("AUGMENTATION")
AUGMENTATION_REGISTRY.__doc__ = """
Data augmentation registery. Registered objects should be subclass of 
`fvcore.transforms.transform.Transform`"""

# Register existing Detectron2 augmentations.
for _, aug_cls in inspect.getmembers(augmentation_impl):
    if type(aug_cls) == type and issubclass(aug_cls, Augmentation):
        AUGMENTATION_REGISTRY.register(aug_cls)

__all__ = [
    "MinMaxNormalize",
]


@AUGMENTATION_REGISTRY.register()
class MinMaxNormalize(Transform):
    """
    Min-Max normalization Transform.

    Returns:
        np.array: image (H, W, 3), 8-bit unsigned integer
    """

    def __init__(self, per_channel: bool = False):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        if self.per_channel:
            return np.dstack(
                [
                    cv2.normalize(img[..., i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    for i in range(img.shape[-1])
                ]
            )
        else:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    def apply_coords(self, coords):
        return coords


def build_augmentation(cfg: CfgNode, train: bool = True) -> list:
    """Build a list of augmentations from config file.

    Args:
        cfg (CfgNode): detectron2 config object
        train (bool, optional): Boolean to indicate whether in training mode. Defaults to True.

    Returns:
        augmentations_list (list): list of augmentations.
    """

    cfg = cfg.clone()
    augmentations_list = []

    aug_cfg_key = cfg.INPUT.AUG if train else cfg.TEST.INPUT.AUG

    if aug_cfg_key:
        for aug in aug_cfg_key:
            name = aug.pop("name")
            try:
                aug = AUGMENTATION_REGISTRY.get(name)(**aug)
            except Exception as e:
                logger.warning(e)
                logger.warning(f"Augmentations in registry: \n{AUGMENTATION_REGISTRY}")

            augmentations_list.append(aug)

    return augmentations_list
