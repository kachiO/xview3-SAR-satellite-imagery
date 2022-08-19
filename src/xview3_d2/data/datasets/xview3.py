"""Functions to create (or load) COCO-format annotations for xView3. """
import json
import logging
import os
import pickle
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, RotatedBoxes
from detectron2.utils.file_io import PathManager
from iopath.common.file_io import file_lock
from xview3_d2.utils import get_rng, get_seed_value, save_dataset

logger = logging.getLogger("detectron2.xview3")

__all__ = [
    "load_xview3_dataset",
    "register_xview3_instances",
    "create_xview3_full_scene_annotations",
    "create_xview3_dataset_dict",
    "THING_CLASSES",
    "CHANNELS",
    "create_xview3_chipped_scene_annotations",
    "create_data_split",
]

THING_CLASSES = ["fishing", "nonfishing", "nonvessel"]
LABEL_MAP = OrderedDict(fishing=0, nonfishing=1, nonvessel=2, background=3)
CHANNELS = [
    "VH_dB",
    "VV_dB",
    "bathymetry",
    "owiWindDirection",
    "owiMask",
    "owiWindQuality",
    "owiWindSpeed",
]
    
# xView3 Constants
NODATA = -32768
# UTM pixel-to-meter conversion value
PIX_TO_M = 10
# Maximum object length
MAX_OBJECT_LENGTH_M = 500

# SAR channels
SAR_CHANNELS = ["VH_dB", "VV_dB"]

# Median pixel intensity ranges for each channel
SCALE_RANGES = {"VH_dB": 44, "VV_dB": 41, "bathymetry": 75}


def load_xview3_dataset(
    dataset_file: Union[os.PathLike, str],
    scene_imagery_dir: Optional[Union[os.PathLike, str]],
) -> list:
    """Load xView3 dataset.

    Args:
        dataset_file (Union[os.PathLike, str]): location of dataset
        scene_imagery_dir (Optional[Union[os.PathLike, str]]): location of scene data. If None, use `dataset_file` parent.

    Returns:
        list of dataset.
    """
    dataset_file = Path(dataset_file)
    with Path(dataset_file).open("rb") as f:
        dataset = pickle.load(f)

    scene_parent = (
        dataset_file.parent if scene_imagery_dir is None else scene_imagery_dir
    )
    scene_parent = Path(scene_parent).resolve()

    for data in dataset:
        data["file_name"] = str(scene_parent / data["file_name"])

    return dataset


def register_xview3_instances(
    name: str,
    dataset_file: Union[os.PathLike, str],
    scene_imagery_dir: Optional[Union[os.PathLike, str]],
):
    """Register xView3 dataset with Detectron2.

    Args:
        name (str): dataset name.
        dataset_file (Union[os.PathLike, str]): location of dataset
        scenes_imagery_root (Optional[Union[os.PathLike, str]]): location of scene data. If None, use `data_root`.
    """
    if name in DatasetCatalog.list():
        logger.warning(f"{name} dataset already in registry. Removing & registering.")
        DatasetCatalog.remove(name)

    DatasetCatalog.register(
        name,
        lambda: load_xview3_dataset(
            Path(dataset_file).absolute(), scene_imagery_dir=scene_imagery_dir
        ),
    )
    MetadataCatalog.get(name).set(thing_classes=THING_CLASSES)
    logger.info(f"{name} dataset registed. {DatasetCatalog.list()[-3:]}")


def create_xview3_full_scene_annotations(
    scene_id: str,
    df_scene_gt: pd.DataFrame,
    df_scene_stats: pd.DataFrame,
    channels: List,
    bbox_width: int = 6,
    bbox_height: int = 8,
) -> dict:
    """
    Create COCO-style annotations for single scene.

    Args:
        scene_id(str): scene id.
        dest_dir(str): destination directory for image chips
        df_scene gt(pd.DataFrame): dataframe consisting of detection labels in scene
        df_scene_stats(pd.DataFrame): dataframe consisting of scene statistics, including pixel min, max, mean, and variance.
        channels(list): list of data channels to include.
        bbox_width(int): bounding box width  to assign to detections if none present in annotations. default 6.
        bbox_height(int): bounding box height to assign to detections if none present in annotations. default 6.

    Returns:
        record (dict): COCO style annotations of all scene image files.
    """

    # Get the height and width of the one of the SAR channels.
    scene_height = df_scene_stats[(df_scene_stats.channel == "VH_dB")]["height"].iloc[0]
    scene_width = df_scene_stats[(df_scene_stats.channel == "VH_dB")]["width"].iloc[0]

    record = {}
    annots_list = []
    record["file_name"] = scene_id
    record["height"] = scene_height
    record["width"] = scene_width
    record["image_id"] = scene_id
    record["scene"] = scene_id
    record["bbox_mode"] = BoxMode.XYWH_ABS
    record["scene_stats"] = (
        df_scene_stats[df_scene_stats.channel.isin(channels)]
        .set_index("channel", drop=True)
        .drop(
            columns=["scene_id", "split"],
        )
        .to_dict()
    )

    for item in df_scene_gt.itertuples():
        _annot = {}

        if item.is_vessel and item.is_fishing:
            _annot["category_id"] = LABEL_MAP["fishing"]
        elif item.is_vessel and not item.is_fishing:
            _annot["category_id"] = LABEL_MAP["nonfishing"]
        elif not item.is_vessel:
            _annot["category_id"] = LABEL_MAP["nonvessel"]
        else:
            _annot["category_id"] = LABEL_MAP["background"]

        _annot["center_xy"] = [item.detect_scene_column, item.detect_scene_row]

        if np.isnan(item.left):
            _annot["bbox"] = [
                item.detect_scene_column - (bbox_width // 2),
                item.detect_scene_row - (bbox_height // 2),
                bbox_width,
                bbox_height,
            ]

        else:
            _annot["bbox"] = [
                item.left,
                item.top,
                item.right - item.left,
                item.bottom - item.top,
            ]

        if len(_annot["bbox"]):
            _annot["bbox"] = [int(b) for b in _annot["bbox"]]

        _annot["bbox_mode"] = BoxMode.XYWH_ABS
        _annot["detection_vessel_length"] = item.vessel_length_m
        _annot["distance_from_shore"] = item.distance_from_shore_km
        _annot["image_id"] = record["image_id"]
        _annot["detect_id"] = item.detect_id
        _annot["detection_source"] = item.source
        _annot["detection_confidence"] = item.confidence
        annots_list.append(_annot)

    record["annotations"] = annots_list

    return record


def get_empty_chip_locations(
    scene_id: str,
    chip_size: int,
    chip_overlap: int,
    scene_height: int,
    scene_width: int,
    df: pd.DataFrame,
    shoreline_dir: Union[os.PathLike, str],
    prop_keep: float,
    seed_rng: Optional[Generator] = None,
) -> pd.DataFrame:
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
        scene_id (str): scene id
        chip_size (int): chip size.
        chip_overlap (int): overlap between adjacent pixels.
        scene_height (int): scene height
        scene_width (int): scene width
        df (pd.DataFrame): dataframe of annotations
        shoreline_dir (Union[os.PathLike, str]): shoreline dataset.
        prop_keep (float): proportion of empty tiles to keep.
        seed_rng (Optional[Generator], optional): random number generator. Defaults to None.

    Returns:
        pd.DataFrame: containing locations of empty chips
    """
    seed_rng = np.random.default_rng(seed_rng)
    stride = chip_size - chip_overlap
    xs = np.arange(0, scene_width - chip_size, stride)
    ys = np.arange(0, scene_height - chip_size, stride)

    def _get_xy(inds, edgex, edgey):
        chips_xs = edgex[inds[:, 1]]
        chips_ys = edgey[inds[:, 0]]
        return (chips_xs, chips_ys)

    def _make_df(xs, ys, shoreline):
        _df = pd.DataFrame(
            columns=["scene_id", "detect_id", "detect_scene_row", "detect_scene_column"]
        )
        _df["detect_scene_row"] = ys + (chip_size // 2)
        _df["detect_scene_column"] = xs + (chip_size // 2)
        _df["scene_id"] = scene_id
        _df["detect_id"] = [f"{scene_id}_row-{y}_col-{x}" for y, x in zip(ys, xs)]
        _df["from_shoreline"] = shoreline
        _df["empty"] = True

        if prop_keep:
            n = min(len(_df), int(prop_keep * len(df)))
            _df = _df.sample(n, random_state=seed_rng.bit_generator, ignore_index=True)
        return _df

    h_annots, (yedges, xedges) = np.histogramdd(
        np.column_stack(
            [
                df.detect_scene_row.values,
                df.detect_scene_column.values,
            ]
        ),
        bins=[ys, xs],
    )

    inds_empty = np.argwhere(h_annots < 1)
    empty_chips_xs, empty_chips_ys = _get_xy(inds_empty, xedges, yedges)
    df_empty_chips = _make_df(empty_chips_xs, empty_chips_ys, shoreline=False)

    if shoreline_dir:
        shoreline_contour_fn = list(Path(shoreline_dir).rglob(f"{scene_id}*.npy"))[0]
        # close to shore data is scene_row, scene_column (y, x).
        shoreline_contour = np.vstack(np.load(shoreline_contour_fn, allow_pickle=True))
        h_shoreline, (yedges, xedges) = np.histogramdd(shoreline_contour, bins=[ys, xs])
        inds_empty_near_shore = np.argwhere((h_annots < 1) & (h_shoreline >= 1))
        inds_empty = np.argwhere((h_annots < 1) & (h_shoreline < 1))
        xs_near_shore, ys_near_shore = _get_xy(inds_empty_near_shore, xedges, yedges)
        df_empty_chips_near_shore = _make_df(
            xs_near_shore, ys_near_shore, shoreline=True
        )
        df_empty_chips = pd.concat(
            [df_empty_chips, df_empty_chips_near_shore], ignore_index=True
        )

    return df_empty_chips


def create_xview3_chipped_scene_annotations(
    scene_id: str,
    chip_size: int,
    df_scene_gt: pd.DataFrame,
    df_scene_stats: pd.DataFrame,
    channels: List,
    bbox_width: int = 6,
    bbox_height: int = 8,
    empty_chip_overlap: int = 32,
    prop_keep_empty: float = 0.2,
    shoreline_dir: Optional[Union[os.PathLike, str]] = None,
    seed_rng: Optional[Generator] = None,
) -> List:
    """
    Create COCO-style annotations for chipped scenes.

    Args:
        scene_tar_file(str):            filename of compressed scene file.
        chip_size(int):                 size of image chips
        dest_dir(str):                  destination directory to save annotations.
        df_scene_gt(pd.DataFrame):      dataframe consisting of detection labels in scene
        df_scene_stats(pd.DataFrame):   dataframe consisting of scene statistics, including pixel min, max, mean, and variance.
        channels(list):                 list of data channels to include.
        bbox_width(int):                width of bounding box to assign to detections if none present in annotations. default 6.
        bbox_height(int):               bounding box height to assign to detections if none present in annotations. default 6.
        seed_rng (Optional[Generator]): random number generator. Default None.

    Returns:
        output_records (List): COCO style annotations of all chipped image files.
    """

    # Get the height and width of the one of the SAR channels.
    scene_height = df_scene_stats[(df_scene_stats.channel == "VH_dB")]["height"].iloc[0]
    scene_width = df_scene_stats[(df_scene_stats.channel == "VH_dB")]["width"].iloc[0]

    # Get empty chips and combine with annotations into single dataframe
    df_empty_chips = get_empty_chip_locations(
        scene_id=scene_id,
        chip_size=chip_size,
        chip_overlap=empty_chip_overlap,
        scene_height=scene_height,
        scene_width=scene_width,
        df=df_scene_gt,
        shoreline_dir=shoreline_dir,
        prop_keep=prop_keep_empty,
        seed_rng=seed_rng,
    )

    # Combine dataframes, annotations and empty chips
    df_scene_gt["empty"] = False
    df_scene_chips = pd.concat([df_scene_gt, df_empty_chips], ignore_index=True)

    chip_half_width = chip_size // 2
    output_records = []

    for counter, sample in enumerate(df_scene_chips.itertuples()):
        record = {}

        fname = f"{scene_id}_chip_{counter + 1:04d}_row-{sample.detect_scene_row}_col-{sample.detect_scene_column}"
        record["file_name"] = fname
        record["height"] = record["width"] = chip_size
        record["image_id"] = Path(fname).stem
        record["scene"] = scene_id
        record["detect_id"] = sample.detect_id
        record["scene_height"] = scene_height
        record["scene_width"] = scene_width
        record["scene_stats"] = (
            df_scene_stats[df_scene_stats.channel.isin(channels)]
            .set_index("channel", drop=True)
            .drop(
                columns=["scene_id", "split"],
            )
            .to_dict()
        )

        chip_center_x, chip_center_y = (
            sample.detect_scene_column,
            sample.detect_scene_row,
        )
        record["scene_center_xy"] = (chip_center_x, chip_center_y)
        chip_top = chip_center_y - chip_half_width
        chip_bottom = chip_center_y + chip_half_width
        chip_left = chip_center_x - chip_half_width
        chip_right = chip_center_x + chip_half_width

        record["scene_chip_coords_xyxy"] = (
            chip_left,
            chip_top,
            chip_right,
            chip_bottom,
        )
        record["scene_chip_top"] = chip_top
        record["scene_chip_left"] = chip_left
        record["vessel_length_m_stats"] = {
            "mean": sample.vessel_length_m_mean,
            "median": sample.vessel_length_m_median,
            "min": sample.vessel_length_m_min,
            "max": sample.vessel_length_m_max,
        }

        record["center_detect_distance_from_shore"] = sample.distance_from_shore_km
        record["bbox_mode"] = "BoxMode.XYWH_ABS"
        # get the detections that are within the chip.
        det_cols = df_scene_chips.detect_scene_column.values
        det_rows = df_scene_chips.detect_scene_row.values
        detection_logic = ((det_cols > chip_left) & (det_cols < chip_right)) & (
            (det_rows > chip_top) & (det_rows < chip_bottom)
        )
        _df = df_scene_chips.iloc[detection_logic]
        annots_list = []

        for item in _df.itertuples():
            if item.empty:
                continue

            new_x = item.detect_scene_column - chip_left
            new_y = item.detect_scene_row - chip_top
            _annot = {}
            _annot["image_id"] = record["image_id"]
            _annot["detect_id"] = item.detect_id
            _annot["center_xy"] = [new_x, new_y]

            if item.is_vessel and item.is_fishing:
                _annot["category_id"] = LABEL_MAP["fishing"]
            elif item.is_vessel and not item.is_fishing:
                _annot["category_id"] = LABEL_MAP["nonfishing"]
            elif not item.is_vessel:
                _annot["category_id"] = LABEL_MAP["nonvessel"]

            if item.split_source == "train":
                _annot["bbox"] = [
                    new_x - (bbox_width // 2),
                    new_y - (bbox_height // 2),
                    bbox_width,
                    bbox_height,
                ]
            elif item.split_source == "val":
                x0 = item.left - chip_left
                y0 = item.top - chip_top
                x1 = item.right - chip_left
                y1 = item.bottom - chip_top

                _annot["bbox"] = [
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                ]
            _annot["bbox_mode"] = "BoxMode.XYWH_ABS"
            _annot["detection_vessel_length"] = item.vessel_length_m
            _annot["distance_from_shore"] = item.distance_from_shore_km
            _annot["detection_source"] = item.source
            _annot["detection_confidence"] = item.confidence

            annots_list.append(_annot)

        record["annotations"] = annots_list
        output_records.append(record)

    return output_records


def convert_to_coco_dict(dataset_name: str):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into COCO json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format

    Adapted from https://github.com/facebookresearch/detectron2/blob/5e38c1f3e6d8e84d3996257b3e7f5d259d06eae6/detectron2/data/datasets/coco.py#L306
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {
            v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
        }
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[
            contiguous_id
        ]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(
                        f"bbox has to be 1-dimensional. Got shape={bbox.shape}."
                    )
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # Computing areas using bounding boxes
            if to_bbox_mode == BoxMode.XYWH_ABS:
                bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                area = Boxes([bbox_xy]).area()[0].item()
            else:
                area = RotatedBoxes([bbox]).area()[0].item()

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            coco_annotation["category_id"] = int(
                reverse_id_mapper(annotation["category_id"])
            )
            coco_annotation["vessel_length_m"] = annotation["detection_vessel_length"]
            coco_annotation["distance_from_shore_km"] = annotation[
                "distance_from_shore"
            ]
            coco_annotation["detect_id"] = annotation["detect_id"]
            coco_annotation["detection_source"] = annotation["detection_source"]
            coco_annotation["confidence"] = annotation["detection_confidence"]

            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "categories": categories,
        "licenses": None,
    }
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def convert_to_coco_json(
    dataset_name: str, output_file: Union[str, os.PathLike], allow_cached=True
):
    """
    Converts xview3 dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion

    Adapted from https://github.com/facebookresearch/detectron2/blob/5e38c1f3e6d8e84d3996257b3e7f5d259d06eae6/detectron2/data/datasets/coco.py#L445
    """

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(
                f"Converting annotations of dataset '{dataset_name}' to COCO format ...)"
            )
            coco_dict = convert_to_coco_dict(dataset_name)

            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)


def create_xview3_dataset_dict(
    gt_labels_csv: Union[str, os.PathLike],
    scene_stats_csv: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    data_channels: List[str],
    bbox_width: int = 6,
    bbox_height: int = 8,
) -> Tuple[Union[str, os.PathLike], List[dict]]:
    """Create Detectron2 dataset dict for all scenes.
    Runs `create_xview3_full_scene_annotations` in a loop for all scenes.

    Args:
        gt_labels_csv (Union[str, os.PathLike]): groundtruth CSV of labels.
        scene_stats_csv (Union[str, os.PathLike]): CSV of scene image statistics.
        data_channels (List[str]): list of channels to include in dataset.
        bbox_width (int): bounding box width to assign to detections if none present in annotations. default 6.
        bbox_height (int): bounding box height to assign to detections if none present in annotations. default 6.
        output_dir (Union[str, os.PathLike]): destination to save dataset.

    Returns:
        dataset_path, dataset (Tuple[Union[str, os.PathLike], List[dict]]): path to saved dataset, dataset_dict
    """
    df_gt = pd.read_csv(gt_labels_csv)
    df_stats = pd.read_csv(scene_stats_csv)

    output_records = []
    scenes = df_gt.scene_id.unique()
    logger.info(f"Creating xView3 detectron2 dataset dict for {len(scenes)}.")

    for scene_id in scenes:
        df_scene_gt = df_gt[df_gt.scene_id == scene_id].reset_index(drop=True)
        df_scene_stats = df_stats[df_stats.scene_id == scene_id].reset_index(drop=True)

        records = create_xview3_full_scene_annotations(
            scene_id=scene_id,
            df_scene_gt=df_scene_gt,
            df_scene_stats=df_scene_stats,
            channels=data_channels,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
        )
        output_records.append(records)

    output_fn = Path(output_dir) / f"{Path(gt_labels_csv).stem}.dataset"
    save_dataset(output_records, output_fn)
    logger.info(f"Saved dataset_dict to {str(output_fn)}")

    return output_fn, output_records


def split_scenes(
    scenes: List[str], p: float = 0.85, seed: Optional[int] = None
) -> Tuple[list, list]:
    """Randomly split scenes into two sets based on provided proportion.

    Args:
        scenes (List[str]): list of scene ids.
        p (float, optional): proportion of scenes to allocate to first set. Defaults to 0.85.
        seed (Optional[int], optional): random seed value. Defaults to None.

    Returns:
        Tuple[list, list]: first set of scenes ids with `p` proportions of scene ids,
                           second set of scene ids with `1 - p` proportions of scene ids.
    """

    rng = get_rng(seed)
    rng.shuffle(scenes)

    num_scenes_split = int(len(scenes) * p)
    scenes_split_1, scenes_split_2 = (
        scenes[:num_scenes_split],
        scenes[num_scenes_split:],
    )
    logger.debug(
        f"Num. scenes in split 1: {len(scenes_split_1)}\n"
        f"Num. scenes in split 2: {len(scenes_split_2)}"
    )

    return scenes_split_1, scenes_split_2


def create_data_split(
    train_labels_csv: str,
    valid_labels_csv: str,
    output_dir: Optional[Union[str, os.PathLike, bool]],
    prop_valid_scenes: float,
    tiny_labels_csv: Optional[Union[str, os.PathLike]] = None,
    exclude_bay_of_biscay: bool = True,
    seed: Optional[int] = None,
):
    """Create new split from merged train and validation splits

    Args:
        train_labels_csv (str): Original train.csv provided in challenge.
        valid_labels_csv (str): Original valid.csv provided in challenge.
        output_dir (Union[str, os.PathLike, bool]): output directory to save new data split.
                    If None, a directory is generated. If False, the data splits are not saved.
        prop_valid_scenes (float): proportion of original validation scenes to include in train split.
        exclude_bay_of_biscay (bool, optional): boolean indicator to exclude train scenes from bay of biscay. Defaults to True.
        tiny_labels_csv (str, os.PathLike, bool]): tiny.csv provided in challenge.
                    If csv provided, filters the tiny.csv file with scenes present in the training set. Default None.
        seed (Optional[int], optional): random seed value. Defaults to None will generate random seed value.
    """
    seed = get_seed_value() if seed is None else seed
    trn_df = pd.read_csv(train_labels_csv)
    val_df = pd.read_csv(valid_labels_csv)
    trn_df["split_source"] = "train"
    val_df["split_source"] = "val"

    trn_included_scenes = trn_df.scene_id.unique().tolist()

    logger.info(f"Num. scenes in training set: {len(trn_included_scenes)}\n"
                f"Num. scenes in validation set: {len(val_df.scene_id.unique())}")

    if exclude_bay_of_biscay:
        logger.info("Excluding Bay of Biscay scenes.")
        exclude_mask = (
            (trn_df.detect_lat.values > 40)
            & (trn_df.detect_lat.values < 52)
            & (trn_df.detect_lon.values > -13)
            & (trn_df.detect_lon.values < 1)
        )
        trn_included_scenes = trn_df.scene_id[~exclude_mask].unique().tolist()

        logger.info(
            f"Excluded {len(trn_df.scene_id[exclude_mask].unique())} scenes from training set. Num. training scenes: {len(trn_included_scenes)}"
        )

    val_scenes_supplement_train, val_scenes_remain_val = split_scenes(
        scenes=val_df.scene_id.unique().tolist(),
        seed=seed,
        p=prop_valid_scenes,
    )

    df_trnvalp = pd.concat(
        [
            trn_df[trn_df.scene_id.isin(trn_included_scenes)],
            val_df[val_df.scene_id.isin(val_scenes_supplement_train)],
        ],
        ignore_index=True,
    )
    # Add vessel length stats, this could be used for imputation of missing vessel lengths.
    df_trnvalp["vessel_length_m_mean"] = df_trnvalp.vessel_length_m.mean()
    df_trnvalp["vessel_length_m_median"] = df_trnvalp.vessel_length_m.median()
    df_trnvalp["vessel_length_m_min"] = df_trnvalp.vessel_length_m.min()
    # max ship value used in metric, based on largest known vessel.
    df_trnvalp["vessel_length_m_max"] = 500

    df_valp = val_df[val_df.scene_id.isin(val_scenes_remain_val)].reset_index(drop=True)
    
    logger.debug(
            f"Num. annotations in train: {len(df_trnvalp)}, Num. scenes: {len(df_trnvalp.scene_id.unique())}\n"
            f"Num. annotations in val: {len(df_valp)}, Num. scenes: {len(df_valp.scene_id.unique())}"
    )
    
    if tiny_labels_csv:
        df_trn_tiny = pd.read_csv(tiny_labels_csv)
        trn_tiny_scenes, val_tiny_scenes = split_scenes(scenes=df_trn_tiny.scene_id.unique().tolist(),seed=seed,p=prop_valid_scenes)
        
        df_trn_tiny = df_trnvalp[
            df_trnvalp.scene_id.isin(trn_tiny_scenes)
        ].reset_index(drop=True)
        
        
        df_val_tiny = df_valp[
            df_valp.scene_id.isin(val_tiny_scenes)
        ]
        
        if df_val_tiny.empty:
            logger.debug(f'')
            rng = get_rng(seed)
            val_scene = rng.choice(df_valp.scene_id.unique())
            df_val_tiny = df_valp[df_valp.scene_id.isin([val_scene])]
            df_val_tiny.reset_index(drop=True, inplace=True)
            
        logger.debug(
            f"#Num. annotations in tiny train: {len(df_trn_tiny)}, Num. scenes: {len(df_trn_tiny.scene_id.unique())}\n"
            f"#Num. annotations in tiny val: {len(df_val_tiny)}, Num. scenes: {len(df_val_tiny.scene_id.unique())}"
        )
        

    if output_dir is None:
        output_dir = (
            Path(train_labels_csv).parent.resolve()
            / f"merge-split-trainval"
            / datetime.now().strftime("%Y%m%d%M%S")
        )
    else:
        output_dir = Path(output_dir)

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        fname = output_dir / "train.csv"
        df_trnvalp.to_csv(fname, index=False)
        logger.info(f"Saved new train (merged train+val) set to: {str(fname)}")

        fname = output_dir / "valid.csv"
        df_valp.to_csv(fname, index=False)
        logger.info(f"Saved new validation set to: {str(fname)}")

        if tiny_labels_csv:
            df_trn_tiny.to_csv(output_dir / "tiny-train.csv", index=False)
            df_val_tiny.to_csv(output_dir / "tiny-valid.csv", index=False)
            
        readme_txt = (
            f"Merge train & validation.\n"
            f"prop valid scenes added to train: {prop_valid_scenes}\n"
            f"random seed: {seed}\n"
            f"Exclude Bay of Biscay scenes in training set: {exclude_bay_of_biscay}"
        )

        (Path(output_dir) / "README.txt").write_text(readme_txt)
        logger.info(readme_txt)

    return df_trnvalp, df_valp
