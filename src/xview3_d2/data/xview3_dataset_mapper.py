import copy
import logging
from typing import List, Union

import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.structures import Boxes, Instances

from .transforms.augmentation import build_augmentation


class xView3DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
    ):

        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] {mode} Augmentations : {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)
        ret = {"is_train": is_train, "augmentations": augs}
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        """Apply transforms to annotations and create Detectron2 Instances.

        Args:
            dataset_dict (_type_): _description_
            transforms (_type_): _description_
            image_shape (_type_): _description_
        """

        if isinstance(transforms, (tuple, list)):
            transforms = T.TransformList(transforms)

        bbox = dataset_dict["bboxes"]
        bbox = transforms.apply_box(bbox).clip(min=0)
        bbox = np.minimum(bbox, np.tile(image_shape, 2)[::-1])
        bbox = Boxes(bbox)
        area_mask = bbox.area() > 0
        instances = Instances(
            image_size=(dataset_dict["height"], dataset_dict["width"])
        )

        instances.gt_boxes = bbox[area_mask]
        instances.gt_classes = torch.tensor(
            (dataset_dict["classes"]), dtype=torch.int64
        )[area_mask]
        instances.gt_lengths = torch.tensor(
            dataset_dict["vessel_lengths"], dtype=torch.float
        )[area_mask]

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = dataset_dict["image"]
        image_shape = (dataset_dict["height"], dataset_dict["width"])

        aug_input = T.AugInput(
            image,
        )
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        if isinstance(image, np.ndarray):
            image = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)).astype("float32")
            )

        dataset_dict["image"] = image
        self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
