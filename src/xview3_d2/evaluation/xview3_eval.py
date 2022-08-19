"""Inference code. 
Adapted code from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py
and https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/evaluator.py
"""
import contextlib
import copy
import io
import itertools
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import detectron2.utils.comm as comm
import numpy as np
import pandas as pd
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate
from torchvision.ops import nms
from xview3_d2.data.datasets.xview3 import MAX_OBJECT_LENGTH_M, convert_to_coco_json

from .metric import score as compute_xview3_score

logger = logging.getLogger("detectron2.xview3Evaluator")

__all__ = ["xView3COCOEvaluator", "xView3F1Evaluator"]


class xView3COCOEvaluator(COCOEvaluator):
    """COCO Evaluator subclass for xView3 dataset.
    Evaluate AP for instance detection/segmentation using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        distributed: bool = True,
        output_dir: Optional[Union[str, os.PathLike]] = None,
        *,
        max_dets_per_image: int = None,
        use_fast_impl: bool = True,
        kpt_oks_sigmas=(),
    ):
        """
        Initialize COCO Evaluator.

        Args:
            dataset_name (str): name of the dataset to be evaluated.
                 It must have either the following corresponding metadata:

                     "json_file": the path to the COCO format annotation

                 Or it must be in detectron2's standard dataset format
                 so it can be converted to COCO format automatically.

            distributed (True): if True, will collect results from all ranks and run evaluation
                 in the main process.
                 Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                 results predicted on the dataset. The dump contains two files:

                 1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                    contains all the results in the format they are produced by the model.
                 2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                 By default in COCO, this limit is to 100, but this can be customized
                 to be greater, as is needed in evaluation metrics AP fixed and AP pool
                 (see https://arxiv.org/pdf/2102.01066.pdf)
                 This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                 Although the results should be very close to the official implementation in COCO
                 API, it is still recommended to compute results with the official API for use in
                 papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
        # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
        # 3rd element (100) is used as the limit on the number of detections per image when
        # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
        # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.

        self._max_dets_per_image = [
            1,
            10,
            100 if max_dets_per_image is None else max_dets_per_image,
        ]
        self._tasks = ("bbox",)
        self._cpu_device = torch.device("cpu")
        self._metadata = MetadataCatalog.get(dataset_name)
        self._do_evaluation = True
        if self._do_evaluation:
            self._kpt_oks_sigmas = kpt_oks_sigmas

        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = str(Path(output_dir) / f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        if not Path(self._metadata.json_file).is_file():
            convert_to_coco_json(dataset_name, self._metadata.json_file)

        json_file = PathManager.get_local_path(self._metadata.json_file)

        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

    def process(self, inputs, outputs):
        """
        Updated function to include length prediction in results with `xview3_instances_to_coco_json`.

        Args:
            inputs:     the inputs to a COCO model (e.g., GeneralizedRCNN).
                        It is a list of dict. Each dict corresponds to an image and
                        contains keys like "height", "width", "file_name", "image_id".
            outputs:    the outputs of a COCO model. It is a list of dicts with key
                        "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = xview3_instances_to_coco_json(
                    instances, input["image_id"]
                )

            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)

            if len(prediction) > 1:
                self._predictions.append(prediction)


class xView3F1Evaluator(DatasetEvaluator):
    """xView3 Competition Metrics Evaluator (F1 scores)"""

    def __init__(
        self,
        dataset_name: str,
        distributed: bool = True,
        iou_thr: float = 0.1,
        score_thr: float = 0.0,
        score_all: bool = False,
        shoreline_dir: Optional[Union[str, os.PathLike]] = None,
        output_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        """Initialize xView3 Competition Metrics Evaluator.

        Args:
            dataset_name (str): _description_
            distributed (bool, optional): _description_. Defaults to True.
            iou_thr (float): IOU threshold to use for Non-Maximum Suppression
            score_all (bool): Boolean to indicate whether to score all confidence values.
                              Default False,scores only HIGH and MEDIUM confidence.
            shoreline_dir (Optional[Union[str, os.PathLike]]): Directory containing shoreline data. Defaults to None.
        """
        self._distributed = distributed
        self._cpu_device = torch.device("cpu")
        self._metadata = MetadataCatalog.get(dataset_name)
        self.iou_thr = iou_thr
        self.score_thr = score_thr
        self.score_all = score_all
        self.shoreline_dir = shoreline_dir
        self._output_dir = output_dir

        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = str(Path(output_dir) / f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        if not Path(self._metadata.json_file).is_file():
            convert_to_coco_json(dataset_name, self._metadata.json_file)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        self.df_gt = self._prepare_gt(json_file)

    def _prepare_gt(self, json_file) -> pd.DataFrame:
        """Prepare groundtruth scene annotations.s

        Args:
            json_file (str): groundtruth annotations.

        Returns:
            (pd.DataFrame): ground truth annotations
        """
        with Path(json_file).open("r") as f:
            data = json.load(f)

        df_gt = pd.DataFrame(data["annotations"])
        df_gt[["x", "y", "w", "h"]] = pd.DataFrame(
            df_gt.bbox.tolist(), index=df_gt.index
        )
        df_gt["is_fishing"] = df_gt.category_id.isin([0])
        df_gt["is_vessel"] = df_gt.category_id.isin([0, 1])

        df_gt["detect_scene_column"] = df_gt.x.values + (df_gt.w.values // 2)
        df_gt["detect_scene_row"] = df_gt.y.values + (df_gt.h.values // 2)

        if not self.score_all:
            df_gt = df_gt[df_gt["confidence"].isin(["HIGH", "MEDIUM"])].reset_index(
                drop=True
            )
        logger.info("Prepared groundtruth dataframe.")
        df_gt.rename(columns={"image_id": "scene_id"}, inplace=True)

        return df_gt

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs:     the inputs to a COCO model (e.g., GeneralizedRCNN).
                        It is a list of dict. Each dict corresponds to an image and
                        contains keys like "height", "width", "file_name", "image_id".
            outputs:    the outputs of a COCO model. It is a list of dicts with key
                        "instances" that contains :class:`Instances`.
        """
        for gt_input, output in zip(inputs, outputs):

            instances = output["instances"].to(self._cpu_device)
            prediction = xview3_instances_to_coco_json(
                instances,
                gt_input["image_id"],
                chip_offset=gt_input.get("chip_offset_xyxy", [0] * 4),
                from_bbox_mode=gt_input.get("bbox_mode", BoxMode.XYXY_ABS),
            )

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        self._results = OrderedDict()
        self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        logger.info("Preparing predictions for scoring.")
        predictions = list(itertools.chain(*predictions))
        df_preds = pd.DataFrame(predictions)
        df_preds[["x", "y", "w", "h"]] = pd.DataFrame(
            df_preds.bbox.tolist(), index=df_preds.index
        )
        df_preds[["chip_x0", "chip_y0", "chip_x1", "chip_y1"]] = pd.DataFrame(
            df_preds.chip_offset.tolist(), index=df_preds.index
        )
        df_preds["is_fishing"] = df_preds.category_id.isin([0])
        df_preds["is_vessel"] = df_preds.category_id.isin([0, 1])

        df_preds["detect_scene_column"] = (df_preds.chip_x0 + df_preds.x.values) + (
            df_preds.w.values // 2
        )
        df_preds["detect_scene_row"] = (df_preds.chip_y0 + df_preds.y.values) + (
            df_preds.h.values // 2
        )
        df_preds["vessel_length_m"] = (
            df_preds.vessel_length_m.values * MAX_OBJECT_LENGTH_M
        )
        df_preds.rename(columns={"image_id": "scene_id"}, inplace=True)

        # discard predictions below threshold.
        logger.info(f"Num. predictions: {len(df_preds)}")
        df_preds = df_preds[df_preds.score > self.score_thr]
        logger.info(
            f"Num. predictions, above threhold of {self.score_thr}: {len(df_preds)}"
        )

        # apply non-maximum suppression with IOU threshold
        df_preds = run_nms_on_df(df_preds, self.iou_thr)

        self._results = compute_xview3_score(
            df_preds,
            self.df_gt,
            distance_tolerance=200,
            shore_tolerance=2,
            shore_root=self.shoreline_dir,
        )
        results_str = tabulate(
            [{k: str(v) for k, v in self._results.items()}],
            tablefmt="pipe",
            headers="keys",
            floatfmt=".4f",
            numalign="left",
        )
        logger.info(f"\n\n[xView3F1Evaluator] results: \n{results_str}")


def _apply_nms_to_predictions(df_in: pd.DataFrame, iou: float) -> pd.DataFrame:
    """Helper function for applying NMS on dataframe.
    Applies NMS per scene.

    Args:
        df_in (pd.DataFrame): Input dataframe.
        iou (float): IOU threshold.

    Returns:
        pd.DataFrame: _description_
    """
    df_pred = df_in.copy().reset_index(drop=True)
    pred_boxes = [
        np.array([box[0], box[1], box[0] + box[2], box[1] + box[2]])
        for box in df_pred.bbox.tolist()
    ]
    scores = torch.as_tensor(df_pred.score.tolist()).float()
    pred_boxes = torch.stack([torch.tensor(x) for x in pred_boxes]).float()

    inds = nms(boxes=pred_boxes, scores=scores, iou_threshold=iou).numpy()
    return df_pred.iloc[inds, :].reset_index(drop=True)


def run_nms_on_df(df: pd.DataFrame, iou: float = 0.1) -> pd.DataFrame:
    """Run NMS on dataframe to remove overlapping detections.
    Args:
        df (pd.DataFrame): input dataframe of detections.
        iou (float, optional): IOU Threshold. Defaults to 0.1.

    Returns:
        pd.DataFrame: Resulting dataframe
    """

    logger.info(f"Num detections pre NMS: {len(df)}")
    dfs = []
    for scene_id in df.scene_id.unique():
        df_in = df[df.scene_id == scene_id]
        dfs.append(_apply_nms_to_predictions(df_in, iou=iou))
    df_out = pd.concat(dfs, ignore_index=True)
    logger.info(f"Num detections post NMS: {len(df_out)}")
    return df_out


def xview3_instances_to_coco_json(
    instances,
    img_id,
    chip_offset=None,
    from_bbox_mode=BoxMode.XYXY_ABS,
):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (str): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, from_bbox_mode, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    lengths = instances.pred_lengths.tolist()

    results = [
        {
            "image_id": img_id,
            "chip_offset": chip_offset,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
            "vessel_length_m": lengths[k],
        }
        for k in range(num_instance)
    ]

    return results
