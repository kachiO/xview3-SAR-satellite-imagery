import ast
import logging
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Generator, List, Optional, Union

import h5py
import numpy as np
import pandas as pd
from detectron2.config import configurable
from detectron2.data import DatasetFromList, MapDataset
from detectron2.data.build import (
    build_batch_data_loader,
    get_detection_dataset_dicts,
    trivial_batch_collator,
)
from detectron2.data.samplers import (
    InferenceSampler,
    RandomSubsetTrainingSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.transforms import CropTransform
from detectron2.structures import BoxMode
from detectron2.utils.logger import _log_api_usage
from sklearn.neighbors import BallTree
from sklearn.utils import compute_sample_weight
from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Sampler
from xview3_d2.data.sampler.sampler import WeightedRandomTrainingSampler
from xview3_d2.data.xview3_dataset_mapper import xView3DatasetMapper
from xview3_d2.utils import load_image_hdf5, load_image_rasterio, load_image_zarr

from .datasets.xview3 import MAX_OBJECT_LENGTH_M, NODATA, SAR_CHANNELS

logger = logging.getLogger("detectron2.xView3Dataset")

__all__ = [
    "xView3FullSceneDataset",
    "xView3ChippedSceneDataset",
    "xview3_build_detection_train_loader",
    "_xview3_train_loader_from_config",
    "_xview3_test_loader_from_config",
    "xview3_build_detection_test_loader",
]


STATS_FUNCS = {
    "min": np.min,
    "max": np.max,
    "median": np.median,
    "mean": np.mean,
}


def capped_randnorm(low, high) -> float:
    """Select a random number from a normal distribution
    between provided low and high value.
    Adapted from 1st place solution: https://github.com/BloodAxe/xView3-The-First-Place-Solution/blob/9a9600e7dfbaa24ff5a72c81061fbbbfed865847/xview3/dataset/random_crop_dataset.py#L16

    Args:
        low (_type_): minimum value
        high (_type_): maximum value

    Returns:
        float: random number between low and high
    """
    mean = random.normalvariate((low + high) * 0.5, sigma=math.sqrt(high - low))
    return np.clip(mean, low, high)


class xView3FullSceneDataset(Dataset):
    """
    xView3 Full Scene Dataset.
    """

    def __init__(
        self,
        dataset: List,
        train: bool,
        serialize_dataset: bool = True,
        channels: List = ["VH_dB", "VV_dB", "bathymetry"],
        chip_size: int = 1280,
        chip_max_jitter: int = 128,
        shoreline_dir: Optional[Union[os.PathLike, str]] = None,
        close_to_shore_distance: int = 2,
        p_choose_detection: float = 0.8,
        balance_near_shore: Optional[str] = "multi",
        balance_crowded_detections: bool = True,
        balance_categories: bool = True,
        chip_overlap: int = 32,
        use_cv2_resize: bool = True,
        p_nodata_pixels: float = 0.8,
        fill_nodata_val: Union[float, str] = 0.0,
        fill_nodata_global: bool = True,
        fill_nan_lengths: Union[float, str] = "mean",
        vessel_length_stats: Optional[dict] = None,
        scene_format: str = "hdf5",
    ):
        """
        xView3 Full Scene Dataset
        Crop from full scene.
        During training: Randomly crop around labeled detections.
        For validation: Crop from fixed grid.

        Args:
            dataset (List): dataset list containing annotations.
            train (bool): boolean flag to indicate if dataset is for training.
            serialize_dataset (bool): whether to hold memory using serialized objects, when
                                      enabled, data loader workers can use shared RAM from master process instead of making a copy.
            channels (List, optional): Image channels to include in dataset. Defaults to ["VH_dB", "VV_dB", "bathymetry"].
            chip_size (int, optional): Size of image chips. Assumes square image. Defaults to 1280.
            chip_max_jitter (int, optional): Maximum pixels to jitter. A random integer
                                             is selected between -chip_max_jitter and +chip_max_jitter.
                                             Defaults to 128.
            shoreline_dir (Optional[Union[os.PathLike, str]], optional): Path to shoreline contour data. Defaults to None.
            close_to_shore_distance (int, optional): Maximum distance (in km) that determines whether an object is close to shore. Defaults to 2.
            p_choose_detection (float, optional): Probability of selecting an image chip with detections. Defaults to 0.8.
            balance_near_shore_type (str, optional): Balance detections close to shore. Defaults to "multi". Options: None, "binary", or "multi"
                                                     If set to "binary", splits the shore distances into two bins [(0, 2), (2, 10000)].
                                                     If set to "multi", splits shore distances into four bins [(0, 1), (1, 2), (2, 10), (10, 10000)].
            balance_crowded_detections (bool, optional): Boolean flag to balance detections that are crowded. Defaults to True.
            balance_categories (bool, optional): Boolean flag to balance categories. Defaults to True.
            chip_overlap (int, optional): Adjacent chip overlap. Used for generating grid for empty chips. Defaults to 32.
            use_cv2_resize (bool): Use OpenCV to resample (resize) ancillary channel. Default True
            p_nodata_pixels (float): Proportion of NODATA pixels above which to exclude. Default 0.8.
            fill_nodata_val (Union[float, str]): Value to fill NODATA pixels. Default 0.0.
                                                 String options: `median`, `mean`, `min`, or `max`,
            fill_nodata_global (bool): If `fill_nodata_val` is a string, use values from global scene statistics otherwise ignored. Defaults True.
            fill_nan_lengths (Union[float, str]): Strategy or value to impute nan vessel lengths.
                                                  Strategy options: `median`, `mean`, or constant value.
                                                  If length stats are not available defaults to length = 0.
            vessel_length_stats (dict): Vessel length statistics. If None provided, the vessel length statistics are computed from the dataset dict. Default None.
            scene_format (str): data format in which scenes are stored. valid options: ["zarr", "hdf5", "tif"]. default hdf5

        """
        self.dataset = dataset
        self._annotations_to_numpy()
        self.dataset = DatasetFromList(self.dataset, serialize=serialize_dataset)
        self.train = train
        self.channels = channels
        self.shoreline_dir = shoreline_dir
        self.chip_size = chip_size
        self.chip_max_jitter = chip_max_jitter
        self.close_to_shore_distance = close_to_shore_distance
        self.p_choose_detection = p_choose_detection
        self.balance_near_shore = balance_near_shore
        self.balance_crowded_detections = balance_crowded_detections
        self.balance_categories = balance_categories
        self.add_empty_chips = self.p_choose_detection == 1
        self.chip_overlap = chip_overlap
        self.use_cv2 = use_cv2_resize
        self.p_nodata = p_nodata_pixels
        self.fill_nodata = fill_nodata_val
        self.fill_nodata_global = fill_nodata_global
        self.fill_nan_lengths = fill_nan_lengths
        self.vessel_length_stats = vessel_length_stats
        self.scene_format = scene_format
        self.scene_reader_func = self._get_image_stack_reader(scene_format)

        if isinstance(self.fill_nodata, str):
            assert self.fill_nodata in [
                "mean",
                "min",
                "max",
                "median",
            ], f"Provided: {self.fill_nodata}. Supported string names: ['mean', 'min', 'max', 'median']"

        if self.train:
            self.impute_lengths_val = self.__get_vessel_length_impute_val()
            self.scene_annot_weights = {}
            self.scene_empty_xy = {}

            for data_dict in self.dataset:
                self._get_annot_weights(data_dict)
                self._get_empty_chip_locations(data_dict)

        else:
            self._get_chip_indices()

    def __get_vessel_length_impute_val(self) -> float:
        """Get the value to impute missing (NaN) vessel lengths.

        Raises:
            ValueError: when data type other than str or float is provided.

        Returns:
            float: value to impute NaN vessel lengths.
        """
        impute_lengths_val = 0
        if isinstance(self.fill_nan_lengths, float):
            impute_lengths_val = self.fill_nan_lengths

        elif isinstance(self.fill_nan_lengths, str):
            assert self.fill_nodata in [
                "mean",
                "median",
            ], f"Provided: {self.fill_nan_lengths}. Supported string names: ['mean', 'median']"

            if self.vessel_length_stats is None:
                logger.info(f"Computing vessel length statistics from training set.")
                vessel_lengths = np.concatenate(
                    [data_dict["vessel_lengths"] for data_dict in self.dataset]
                )

                self.vessel_length_stats = {
                    "min": np.nanmin(vessel_lengths),
                    "mean": np.nanmean(vessel_lengths),
                    "median": np.nanmedian(vessel_lengths),
                }

            impute_lengths_val = self.vessel_length_stats[self.fill_nan_lengths]

        else:
            raise ValueError(
                f"Expected str or float for fill_nan_lengths, got {type(self.fill_nan_lengths)}"
            )
        logger.info(f"Impute NaN vessel lengths with {impute_lengths_val}")

        return impute_lengths_val

    def __len__(self) -> int:
        if self.train:
            # return number of dataset/scenes. allows workers to open different scene files.
            return len(self.dataset)
        else:
            # return number of chips
            return self._len

    def __len_annotations__(self) -> int:
        if self.train:
            return sum([len(v) for _, v in self.scene_annot_weights.items()])

        else:
            # return number of chips
            return self._len

    def _annotations_to_numpy(
        self,
    ):
        """Convert annotations in data dictionary into numpy instead of list."""
        for data_dict in self.dataset:
            df = pd.DataFrame(data_dict["annotations"])
            data_dict["center_xy"] = np.stack(df["center_xy"])
            data_dict["bboxes"] = np.stack(df["bbox"])
            data_dict["classes"] = df["category_id"].to_numpy(int)
            data_dict["vessel_lengths"] = df["detection_vessel_length"].to_numpy(float)
            data_dict["annotation_confidence"] = df["detection_confidence"].to_numpy()
            data_dict["distance_from_shore"] = df["distance_from_shore"].to_numpy(float)
            data_dict["detect_id"] = df["detect_id"].to_numpy()

    def _get_annot_weights(self, data_dict):
        """
        Compute weights for each annotation in order
        to balance different aspects of the labeled data,
        including balancing near shore detections, crowded detections,
        and class labels.
        Adapted from 1st place solution:
            https://github.com/BloodAxe/xView3-The-First-Place-Solution/blob/9a9600e7dfbaa24ff5a72c81061fbbbfed865847/xview3/dataset/random_crop_dataset.py#L86

        Args:
            data_dict (_type_): _description_
        """
        scene_id = data_dict["image_id"]
        df = pd.DataFrame(data_dict["annotations"])
        self.scene_annot_weights[scene_id] = np.ones(len(df), dtype=np.float32)

        if self.balance_near_shore == "multi":
            shore_bins = pd.cut(
                df.distance_from_shore.values,
                bins=[0, 1, 2, 10, 10000],
                labels=[1, 2, 10, 10000],
            ).tolist()
        else:
            shore_bins = df.distance_from_shore.values < self.close_to_shore_distance

        balance_counter = 0

        if self.balance_near_shore and len(np.bincount(shore_bins)):
            shore_weights = compute_sample_weight("balanced", shore_bins)
            shore_weights /= shore_weights.sum()
            self.scene_annot_weights[scene_id] += shore_weights
            balance_counter += 1

        if self.balance_crowded_detections:
            centers = np.stack(df["center_xy"])
            tree = BallTree(centers, leaf_size=8, metric="chebyshev")
            neighbors_count = tree.query_radius(
                centers, r=self.chip_size // 2, count_only=True
            )
            crowd_weights = np.reciprocal(neighbors_count.astype(np.float32))
            crowd_weights /= crowd_weights.sum()
            self.scene_annot_weights[scene_id] += crowd_weights
            balance_counter += 1

        if self.balance_categories:
            category_weights = compute_sample_weight(
                "balanced", df["category_id"].values
            )
            self.scene_annot_weights[scene_id] += category_weights
            balance_counter += 1

        if balance_counter:
            self.scene_annot_weights[scene_id] /= balance_counter

    def _generate_xy_coords(self, data_dict: dict) -> Generator:
        """Get initial XY coordinates for image chip.

        Args:
            data_dict (dict): dataset dictionary

        Yields:
            Generator: yields initial x0, y0 coordinates for fixed grid.
        """
        stride = self.chip_size - self.chip_overlap
        for xs in np.arange(0, data_dict["width"] - self.chip_size, stride):
            for ys in np.arange(0, data_dict["height"] - self.chip_size, stride):
                yield xs, ys

    def _get_chip_indices(self):
        """Map image chips to indices."""
        self.inds_dataset_crop = []

        for idx, data in enumerate(self.dataset):
            for xy in self._generate_xy_coords(data):
                self.inds_dataset_crop.append([idx, xy])
        self._len = len(self.inds_dataset_crop)

    def _get_empty_chip_locations(self, data_dict: dict):
        """
        Get empty chip locations.

        Creates a 2D histogram of the x,y coordinates based on labeled detections in the scene.
        The 2D histogram is a m x n grid, where each cell on the grid represents
        chip_size x chip_size location on the scene, and adjacent cells share an overlap of
        chip_overlap pixels.

        The 2D histogram is used to create a mask that identifies potentially empty
        locations, i.e. without annotated detections. If shoreline data is available, the same
        approach is used to identify empty locations that are close to shore.

        Args:
            data_dict(dict): dataset dictionary

        """

        scene_id = data_dict["image_id"]
        stride = self.chip_size - self.chip_overlap
        xs = np.arange(0, data_dict["width"] - self.chip_size, stride)
        ys = np.arange(0, data_dict["height"] - self.chip_size, stride)
        df = pd.DataFrame(data_dict["annotations"])

        def _get_xy(inds, edgex, edgey):
            chips_xs = edgex[inds[:, 1]]
            chips_ys = edgey[inds[:, 0]]
            xy = np.column_stack((chips_xs, chips_ys))
            return xy

        # swap to scene_row, scene_column (y, x).
        h_annots, (yedges, xedges) = np.histogramdd(
            np.stack(df["center_xy"])[:, [1, 0]], bins=[ys, xs]
        )

        inds_empty = np.argwhere(h_annots < 1)
        empty_xy = _get_xy(inds_empty, xedges, yedges)
        empty_xy_near_shore = None

        if self.shoreline_dir:
            shoreline_contour_fn = list(
                Path(self.shoreline_dir).rglob(f"{scene_id}*.npy")
            )[0]
            # close to shore data is scene_row, scene_column (y, x).
            shoreline_contour = np.vstack(
                np.load(shoreline_contour_fn, allow_pickle=True)
            )
            h_shoreline, (yedges, xedges) = np.histogramdd(
                shoreline_contour, bins=[ys, xs]
            )
            inds_empty_near_shore = np.argwhere((h_annots < 1) & (h_shoreline >= 1))
            inds_empty = np.argwhere((h_annots < 1) & (h_shoreline < 1))
            empty_xy = _get_xy(inds_empty, xedges, yedges)
            empty_xy_near_shore = _get_xy(inds_empty_near_shore, xedges, yedges)

        self.scene_empty_xy[scene_id] = (empty_xy, empty_xy_near_shore)

    def _get_image_stack_reader(
        self,
        format: str,
    ):
        """Get function to read scene data.

        Args:
            format (str): format of input scene.
        """

        if format == "hdf5":
            return load_image_hdf5
        elif format == "tif":
            return load_image_rasterio
        elif format == "zarr":
            return load_image_zarr
        else:
            raise ValueError(f"Unknown format: {format}")

    def __fill_nodata_pixels(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        channel: str,
        scene_stats: Optional[dict] = None,
    ) -> np.ndarray:
        """Fill NODATA pixels

        Args:
            img (np.ndarray): image data.
            mask (np.ndarray): boolean mask indicating NODATA pixels
            channel (str): data channel of image.
            scene_stats (Optional[dict], optional): Scene level statistics. Defaults to None.

        Returns:
            img (np.ndarray): image array filled with
        """
        if isinstance(self.fill_nodata, float):
            img[mask] = self.fill_nodata
            return img

        elif isinstance(self.fill_nodata, str):
            if self.fill_nodata_global and scene_stats:
                fill_value = scene_stats[self.fill_nodata][channel]
            else:
                fill_value = STATS_FUNCS[self.fill_nodata](img[~mask])

            img[mask] = fill_value
        else:
            raise ValueError(f"Unknown fill_nodata= {self.fill_nodata}")
        return img

    def __getitem__(self, idx: int):
        """Get dataset item.

        Args:
            idx (int): integer of dataset item to select.
        """
        if self.train:
            data_dict = self.dataset[idx]
            scene_id = data_dict["image_id"]

            if random.random() < self.p_choose_detection:
                # sample from annotated detections.
                weights = self.scene_annot_weights[scene_id]
                annot_ind = random.choices(range(len(weights)), weights)[0]
                center_x, center_y = data_dict["center_xy"][annot_ind]
                dx = int(capped_randnorm(-self.chip_max_jitter, self.chip_max_jitter))
                dy = int(capped_randnorm(-self.chip_max_jitter, self.chip_max_jitter))
                x0 = max(0, center_x + dx - self.chip_size // 2)
                y0 = max(0, center_y + dy - self.chip_size // 2)

            else:
                # sample from empty location
                if (
                    isinstance(self.scene_empty_xy[scene_id][1], np.ndarray)
                    and random.random() < 0.5
                ):
                    # near shore
                    empty_xy = self.scene_empty_xy[scene_id][1]
                else:
                    # far from shore
                    empty_xy = self.scene_empty_xy[scene_id][0]
                ind = random.sample(range(len(empty_xy)), 1)[0]
                x0, y0 = empty_xy[ind]
        else:
            dataset_ind, (x0, y0) = self.inds_dataset_crop[idx]
            data_dict = self.dataset[dataset_ind]

        max_scene_width = data_dict["width"]
        max_scene_height = data_dict["height"]
        x1 = min(max_scene_width, x0 + self.chip_size)
        y1 = min(max_scene_height, y0 + self.chip_size)

        fname = Path(data_dict["file_name"]).with_suffix('.h5')
        img_data = h5py.File(fname, 'r')
        image_stack_store = {chan: img_data[chan][y0:y1, x0:x1] for chan in set(self.channels)}
    
        # If the proportion of NODATA pixels exceeds the threshold of self.p_nodata, retrieve another crop.
        sar_chan = [ch for ch in SAR_CHANNELS if ch in image_stack_store.keys()][0]
        mask_nodata = image_stack_store[sar_chan] == NODATA
        prop_nodata = (mask_nodata).sum() / np.prod(image_stack_store[sar_chan].shape)
        if prop_nodata > self.p_nodata and self.train:
            self[idx]

        if (mask_nodata).sum():
            image_stack_store = {
                chan: self.__fill_nodata_pixels(
                    img,
                    scene_stats=data_dict["scene_stats"],
                    channel=chan,
                    mask=mask_nodata,
                )
                for chan, img in image_stack_store.items()
            }

        crop_transform = CropTransform(
            x0, y0, self.chip_size, self.chip_size, max_scene_width, max_scene_height
        )
        from_mode = eval(data_dict["bbox_mode"]) if isinstance(data_dict["bbox_mode"], str) else data_dict["bbox_mode"]
        bboxes = BoxMode.convert(
            data_dict["bboxes"], from_mode, BoxMode.XYXY_ABS
        )
        bboxes = crop_transform.apply_box(np.array(bboxes)).clip(min=0)

        image_stack = np.dstack([image_stack_store[chan] for chan in self.channels])
        image_shape = image_stack.shape[:2]
        bboxes = np.minimum(bboxes, np.tile(image_shape, 2)[::-1])
        bbox_mask = np.all(bboxes[:, :2] != bboxes[:, 2:], axis=1)

        vessel_lengths = data_dict["vessel_lengths"][bbox_mask]

        if self.train:
            vessel_lengths = np.nan_to_num(vessel_lengths, nan=self.impute_lengths_val)
            vessel_lengths /= MAX_OBJECT_LENGTH_M

        output_data_dict = {
            "image_id": data_dict["image_id"],
            "height": image_shape[0],
            "width": image_shape[1],
            "detect_id": data_dict["detect_id"][bbox_mask],
            "image": image_stack,
            "bboxes": bboxes[bbox_mask],
            "classes": data_dict["classes"][bbox_mask],
            "vessel_lengths": vessel_lengths,
            "confidence": data_dict["annotation_confidence"][bbox_mask],
            "bbox_mode": BoxMode.XYXY_ABS,
            "distance_from_shore": data_dict["distance_from_shore"][bbox_mask],
            "chip_offset_xyxy": (x0, y0, x1, y1),
        }
        return output_data_dict


class xView3ChippedSceneDataset(Dataset):
    """
    xView3 Chipped Scenes Dataset.
    """

    def __init__(
        self,
        dataset: List,
        train: bool = True,
        serialize_dataset: bool = True,
        channels: List = ["VH_dB", "VV_dB", "bathymetry"],
        chip_size: int = 1024,
        chip_max_jitter: int = 128,
        p_choose_detection: float = 0.8,
        balance_near_shore: Optional[str] = "multi",
        balance_categories: bool = True,
        repeat_factor: float = 0.0,
        fill_nodata_val: Union[float, str] = "mean",
        fill_nodata_global: bool = True,
        fill_nan_lengths: Union[float, str] = "mean",
        vessel_length_stats: Optional[dict] = None,
    ):
        """
        Args:
            dataset (List): dataset list containing annotations.
            serialize_dataset (bool): whether to hold memory using serialized objects, when
                                      enabled, data loader workers can use shared RAM from master process instead of making a copy.
            train (bool): Boolean flag to indicate training dataset.
            channels (List, optional): Image channels to include in dataset. Defaults to ["VH_dB", "VV_dB", "bathymetry"].
            chip_size (int, optional): Size of image chips. Assumes square image. Defaults to 1280.
            chip_max_jitter (int, optional): Maximum pixels to jitter. A random integer
                                             is selected between -chip_max_jitter and +chip_max_jitter.
                                             Defaults to 128.
            shoreline_dir (Optional[Union[os.PathLike, str]], optional): Path to shoreline contour data. Defaults to None.
            p_choose_detection (float, optional): Probability of selecting an image chip with detections. Defaults to 0.8.
            balance_near_shore_type (str, optional): Balance detections close to shore. Defaults to "multi". Options: None, "binary", or "multi"
                                                     If set to "binary", splits the shore distances into two bins [(0, 2), (2, 10000)].
                                                     If set to "multi", splits shore distances into four bins [(0, 1), (1, 2), (2, 10), (10, 10000)].
            balance_categories (bool, optional): Boolean flag to balance categories. Defaults to True.
            fill_nodata_val (Union[float, str]): Strategy or constant value to fill (impute) NODATA pixels. Default "mean".
                                                 Strategy options: `median`, `mean`, `min`, `max`, or constant value.
            fill_nodata_global (bool): If `fill_nodata_val` is a string, use values from global scene statistics otherwise ignored. Defaults True.
            fill_nan_lengths (Union[float, str]): Strategy or value to impute nan vessel lengths.
                                                  Strategy options: `median`, `mean`, or constant value.
                                                  If length stats are not available defaults to length = 0.
            vessel_length_stats (dict): Vessel length statistics. If None provided, the vessel length statistics are computed from the dataset dict. Default None.
        """

        self.dataset = dataset
        self.train = train
        self._annotations_to_numpy()
        self.dataset = DatasetFromList(self.dataset, serialize=serialize_dataset)
        self.channels = channels
        self.chip_size = chip_size
        self.chip_max_jitter = chip_max_jitter
        self.balance_near_shore = balance_near_shore
        self.balance_categories = balance_categories
        self.use_repeat_factor = repeat_factor
        self.prop_empty = 1 - p_choose_detection
        self.fill_nodata = fill_nodata_val
        self.fill_nodata_global = fill_nodata_global
        self.fill_nan_lengths = fill_nan_lengths
        self.vessel_length_stats = vessel_length_stats

        if isinstance(self.fill_nodata, str):
            assert self.fill_nodata in [
                "mean",
                "min",
                "max",
                "median",
            ], f"Provided: {self.fill_nodata}. Supported string names: ['mean', 'min', 'max', 'median']"

        self.weights = None

        if self.train:
            self.impute_lengths_val = self.__get_vessel_length_impute_val()

            if self.balance_categories or self.balance_near_shore:
                self.__compute_weights()

    def __get_vessel_length_impute_val(self) -> float:
        """Get the value to impute missing (NaN) vessel lengths.

        Raises:
            ValueError: when data type other than str or float is provided.

        Returns:
            float: value to impute NaN vessel lengths.
        """
        if isinstance(self.fill_nan_lengths, str):
            assert self.fill_nodata in [
                "mean",
                "median",
            ], f"Provided: {self.fill_nan_lengths}. Supported string names: ['mean', 'median']"
            if self.vessel_length_stats is None:
                # Use precomputed vessel lengths provided in dataset.
                self.vessel_length_stats = next(
                    (
                        ds["vessel_length_m_stats"]
                        for ds in self.dataset
                        if ~ds["empty"] and "vessel_length_m_stats" in ds.keys()
                    ),
                    None,
                )
            impute_lengths_val = self.vessel_length_stats[self.fill_nan_lengths]
        elif isinstance(self.fill_nan_lengths, float):
            impute_lengths_val = self.fill_nan_lengths
        else:
            raise ValueError(
                f"Expected str or float for fill_nan_lengths, got {type(self.fill_nan_lengths)}"
            )

        logger.info(f"Impute NaN vessel lengths with {impute_lengths_val}")
        return impute_lengths_val

    def _annotations_to_numpy(
        self,
    ):
        """Convert annotations in data dictionary into numpy instead of list."""
        for data_dict in self.dataset:
            has_annot = True if len(data_dict["annotations"]) else False

            df = pd.DataFrame(data_dict["annotations"])
            data_dict["bboxes"] = (
                np.stack(df["bbox"]) if has_annot else np.empty((0, 4), dtype=np.int64)
            )
            data_dict["classes"] = (
                df["category_id"].to_numpy(int)
                if has_annot
                else np.empty(0, dtype=np.int64)
            )
            data_dict["vessel_lengths"] = (
                df["detection_vessel_length"].to_numpy(np.float32)
                if has_annot
                else np.empty(0, dtype=np.float32)
            )
            data_dict["annotation_confidence"] = (
                df["detection_confidence"].to_numpy(str)
                if has_annot
                else np.empty(0, dtype=str)
            )
            data_dict["distance_from_shore"] = (
                df["distance_from_shore"].to_numpy(float)
                if has_annot
                else np.empty(0, dtype=np.float32)
            )
            data_dict["ann_detect_id"] = (
                df["detect_id"].to_numpy().astype(str)
                if has_annot
                else np.empty(0, dtype=str)
            )
            data_dict["empty"] = ~has_annot

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.dataset)

    def __impute_nodata_pixels(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        channel: str,
        scene_stats: Optional[dict] = None,
    ) -> np.ndarray:
        """Impute NODATA pixels

        Args:
            img (np.ndarray): image data.
            mask (np.ndarray): boolean mask indicating NODATA pixels
            channel (str): data channel of image.
            scene_stats (Optional[dict], optional): Scene level statistics. Defaults to None.

        Returns:
            np.ndarray: image
        """
        if isinstance(self.fill_nodata, float):
            img[mask] = self.fill_nodata
            return img

        elif isinstance(self.fill_nodata, str):
            if self.fill_nodata_global and scene_stats:
                fill_value = scene_stats[self.fill_nodata][channel]
            else:
                fill_value = STATS_FUNCS[self.fill_nodata](img[~mask])

            img[mask] = fill_value
        else:
            raise ValueError(f"Unknown fill_nodata= {self.fill_nodata}")
        return img

    def __compute_weights(self):
        """Compute weights for each element in dataset."""

        self.weights = np.ones(len(self.dataset), dtype=np.float32)

        if self.balance_near_shore:
            distance_from_shore = [
                ds["center_detect_distance_from_shore"] for ds in self.dataset
            ]
            bins = [-10, 0, 2, 10000]
            labels = [-1, 2, 10000]

            if self.balance_near_shore == "multi":
                bins = [-10, 0, 1, 2, 10, 10000]
                labels = [-1, 1, 2, 10, 10000]

            shore_bins = pd.cut(distance_from_shore, bins=bins, labels=labels).tolist()
            nan_indices = np.isnan(shore_bins)
            shore_bins = np.nan_to_num(shore_bins, nan=-1)
            shore_weights = compute_sample_weight(
                "balanced", shore_bins, indices=np.where(~nan_indices)[0]
            )
            shore_weights[nan_indices] = (
                np.unique(shore_weights).sum() * self.prop_empty
            )
            shore_weights /= shore_weights.sum()
            self.weights = np.prod(
                [self.weights, shore_weights], axis=0, dtype=np.float64
            )

        if self.balance_categories:
            center_category_ids = []
            for d in self.dataset:
                cat_id = d["classes"][np.where(d["ann_detect_id"] == d["detect_id"])[0]]
                if cat_id.size == 0:
                    cat_id = np.nan
                else:
                    cat_id = int(cat_id)
                center_category_ids.append(cat_id)

            class_weight = "balanced"
            nan_indices = np.isnan(center_category_ids)
            center_category_ids = np.nan_to_num(center_category_ids, nan=-1)
            indices = np.where(~nan_indices)[0]

            if self.use_repeat_factor:
                category_freq = Counter(center_category_ids)
                n = sum(category_freq.values())
                class_weight = {
                    cat_id: max(
                        1.0, math.sqrt(self.use_repeat_factor / (cat_freq_count / n))
                    )
                    for cat_id, cat_freq_count in category_freq.items()
                }
                indices = None

            category_weights = compute_sample_weight(
                class_weight, center_category_ids, indices=indices
            )
            category_weights[nan_indices] = (
                np.unique(category_weights[~nan_indices]).sum() * self.prop_empty
            )
            category_weights /= category_weights.sum()
            self.weights = np.prod(
                [self.weights, category_weights], axis=0, dtype=np.float64
            )

    def __getitem__(self, idx):
        """Retrieve dataset element."""
        data_dict = self.dataset[idx]
        fn = Path(data_dict["file_name"])

        if not len(fn.suffix):
            fn = fn.with_suffix('.npz')
        _image_stack = np.load(fn)
        image_stack = {f:_image_stack[f] for f in _image_stack.files}

        sar_chan = [ch for ch in SAR_CHANNELS if ch in image_stack.keys()][0]
        mask_nodata = image_stack[sar_chan] == NODATA

        if (mask_nodata).sum():
            image_stack = {
                chan: self.__impute_nodata_pixels(
                    img,
                    scene_stats=data_dict["scene_stats"],
                    channel=chan,
                    mask=mask_nodata,
                )
                for chan, img in image_stack.items()
            }

        center_x, center_y = data_dict["scene_center_xy"]
        center_x -= data_dict["scene_chip_left"]
        center_y -= data_dict["scene_chip_top"]

        dx = int(capped_randnorm(-self.chip_max_jitter, self.chip_max_jitter))
        dy = int(capped_randnorm(-self.chip_max_jitter, self.chip_max_jitter))
        x0 = max(0, center_x + dx - self.chip_size // 2)
        y0 = max(0, center_y + dy - self.chip_size // 2)
        x1 = min(data_dict["width"], x0 + self.chip_size)
        y1 = min(data_dict["height"], y0 + self.chip_size)

        crop_transform = CropTransform(
            x0=x0,
            y0=y0,
            w=self.chip_size,
            h=self.chip_size,
            orig_w=data_dict["width"],
            orig_h=data_dict["height"],
        )
        bbox_mode = (
            eval(data_dict["bbox_mode"])
            if isinstance(data_dict["bbox_mode"], str)
            else data_dict["bbox_mode"]
        )
        bboxes = BoxMode.convert(data_dict["bboxes"], bbox_mode, BoxMode.XYXY_ABS)
        bboxes = crop_transform.apply_box(np.array(bboxes)).clip(min=0)

        image_stack = np.dstack(
            [image_stack[chan][y0:y1, x0:x1] for chan in self.channels]
        )
        image_shape = image_stack.shape[:2]
        bboxes = np.minimum(bboxes, np.tile(image_shape, 2)[::-1])
        bbox_mask = np.all(bboxes[:, :2] != bboxes[:, 2:], axis=1)
        vessel_lengths = data_dict["vessel_lengths"][bbox_mask]

        if self.train:
            vessel_lengths = np.nan_to_num(vessel_lengths, nan=self.impute_lengths_val)
            vessel_lengths /= MAX_OBJECT_LENGTH_M

        output_data_dict = {
            "image_id": data_dict["image_id"],
            "scene_id": data_dict["scene"],
            "height": image_shape[0],
            "width": image_shape[1],
            "detect_id": data_dict["ann_detect_id"][bbox_mask],
            "image": image_stack,
            "bboxes": bboxes[bbox_mask],
            "classes": data_dict["classes"][bbox_mask],
            "vessel_lengths": vessel_lengths,
            "confidence": data_dict["annotation_confidence"][bbox_mask],
            "bbox_mode": BoxMode.XYXY_ABS,
            "distance_from_shore": data_dict["distance_from_shore"][bbox_mask],
            "chip_offset_xyxy": (x0, y0, x1, y1),
        }
        return output_data_dict


def _xview3_train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = xView3DatasetMapper(cfg, is_train=True)

    dataset_name = cfg.DATALOADER.DATASET_TRAIN
    if dataset_name == "xView3FullSceneDataset":
        trn_dataset = xView3FullSceneDataset(
            dataset=dataset,
            train=True,
            channels=cfg.INPUT.DATA.CHANNEL_NAMES,
            chip_size=cfg.INPUT.DATA.CHIP_SIZE,
            chip_max_jitter=cfg.INPUT.DATA.CHIP_MAX_JITTER,
            shoreline_dir=cfg.INPUT.DATA.SHORELINE_DIR,
            close_to_shore_distance=cfg.INPUT.DATA.CLOSE_TO_SHORE_KM,
            p_choose_detection=cfg.INPUT.DATA.PROP_DETECTIONS,
            balance_near_shore=cfg.INPUT.DATA.BALANCE_NEAR_SHORE,
            balance_crowded_detections=cfg.INPUT.DATA.BALANCE_CROWDED,
            balance_categories=cfg.INPUT.DATA.BALANCE_CATEGORIES,
            chip_overlap=cfg.INPUT.DATA.CHIP_OVERLAP,
            use_cv2_resize=cfg.INPUT.DATA.CV2_RESIZE,
            p_nodata_pixels=cfg.INPUT.DATA.PROP_NODATA_PIXELS,
            fill_nodata_val=cfg.INPUT.DATA.FILL_NODATA_VAL,
            fill_nodata_global=cfg.INPUT.DATA.FILL_NODATA_GLOBAL,
            fill_nan_lengths=cfg.INPUT.DATA.LENGTHS_FILL_NAN,
            vessel_length_stats=ast.literal_eval(cfg.INPUT.DATA.TRAIN_LENGTH_STATS),
            scene_format=cfg.INPUT.SCENE_FORMAT,
        )
    elif dataset_name == "xView3ChippedSceneDataset":
        trn_dataset = xView3ChippedSceneDataset(
            dataset=dataset,
            channels=cfg.INPUT.DATA.CHANNEL_NAMES,
            chip_size=cfg.INPUT.DATA.CHIP_SIZE,
            chip_max_jitter=cfg.INPUT.DATA.CHIP_MAX_JITTER,
            p_choose_detection=cfg.INPUT.DATA.PROP_DETECTIONS,
            balance_near_shore=cfg.INPUT.DATA.BALANCE_NEAR_SHORE,
            balance_categories=cfg.INPUT.DATA.BALANCE_CATEGORIES,
            fill_nodata_val=cfg.INPUT.DATA.FILL_NODATA_VAL,
            fill_nodata_global=cfg.INPUT.DATA.FILL_NODATA_GLOBAL,
            fill_nan_lengths=cfg.INPUT.DATA.LENGTHS_FILL_NAN,
            vessel_length_stats=ast.literal_eval(cfg.INPUT.DATA.TRAIN_LENGTH_STATS),
        )
    else:
        raise ValueError(f"Unknown training dataset: {dataset_name}")

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info(f"Using training sampler {sampler_name}")

        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(trn_dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = (
                RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    trn_dataset, cfg.DATALOADER.REPEAT_THRESHOLD
                )
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        elif sampler_name == "RandomSubsetTrainingSampler":
            sampler = RandomSubsetTrainingSampler(
                len(trn_dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO
            )
        elif sampler_name == "WeightedRandomTrainingSampler":
            sampler = WeightedRandomTrainingSampler(
                size=len(dataset), weights=trn_dataset.weights
            )
        else:
            raise ValueError(f"Unknown training sampler: {sampler_name}")

    logger.info(f"Dataset: {dataset_name}. \nSampler: {sampler_name}")

    return {
        "dataset": trn_dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_xview3_train_loader_from_config)
def xview3_build_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    if isinstance(dataset, IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _xview3_test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset_list = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)]
            for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = xView3DatasetMapper(cfg, is_train=False)

    dataset = xView3FullSceneDataset(
        dataset=dataset_list,
        train=False,
        serialize_dataset=True,
        channels=cfg.INPUT.DATA.CHANNEL_NAMES,
        chip_size=cfg.INPUT.DATA.CHIP_SIZE,
        shoreline_dir=cfg.TEST.INPUT.SHORELINE_DIR,
        close_to_shore_distance=cfg.TEST.INPUT.CLOSE_TO_SHORE_KM,
        p_choose_detection=False,
        balance_near_shore=False,
        balance_crowded_detections=False,
        balance_categories=False,
        chip_overlap=cfg.TEST.INPUT.CHIP_OVERLAP,
        use_cv2_resize=cfg.INPUT.DATA.CV2_RESIZE,
        p_nodata_pixels=1.0,
        fill_nodata_val=cfg.INPUT.DATA.FILL_NODATA_VAL,
        fill_nodata_global=cfg.INPUT.DATA.FILL_NODATA_GLOBAL,
        scene_format=cfg.TEST.INPUT.SCENE_FORMAT,
    )
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "batch_size": cfg.TEST.IMS_PER_BATCH,
    }


@configurable(from_config=_xview3_test_loader_from_config)
def xview3_build_detection_test_loader(dataset, *, mapper, batch_size=1, num_workers=0):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    data_loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader
