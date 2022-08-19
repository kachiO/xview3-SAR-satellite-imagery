"""Create detectron2 dataset dict for xView3."""
import ast
import itertools
import logging
import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Callable, List, Tuple, Union

import pandas as pd
from xview3_d2.data.datasets.xview3 import (
    CHANNELS,
    create_data_split,
    create_xview3_chipped_scene_annotations,
    create_xview3_full_scene_annotations,
)
from xview3_d2.utils import configure_logging, save_dataset

logger = logging.getLogger("Create-xview3-dataset")


def parse_args():
    """Parse commandline arguments."""
    parser = ArgumentParser(description="Save xView3 imagery to File Storage format.")
    parser.add_argument(
        "--dataset-type",
        default="full",
        choices=["full", "chipped"],
        type=str,
        help="Type of xview3 dataset dictionary to create, full scenes or chipped scenes.",
    )
    parser.add_argument(
        "--train-labels-csv", type=str, help="Groundtruth train labels csv."
    )
    parser.add_argument(
        "--valid-labels-csv", type=str, help="Groundtruth validation labels csv."
    )
    parser.add_argument(
        "--tiny-labels-csv",
        default=None,
        type=str,
        help="Groundtruth tiny subset labels csv.",
    )
    parser.add_argument(
        "--prop-valid-scenes",
        default=0.85,
        type=float,
        help="Proportion of validation scenes to include in training set. Default 0.85",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Random seed value. Default None, will generate a random seed.",
    )
    parser.add_argument(
        "--exclude-bay-of-biscay",
        default=True,
        help="Exclude scenes from Bay of Biscay from training set, which is not present in validation set. ",
    )
    parser.add_argument(
        "--output-dir",
        default="/opt/ml/processing/output/data/prepared/",
        help="Directory to save split data.",
    )
    parser.add_argument(
        "--gt-labels-dir",
        default=None,
        help="Directory of groundtruth labels. Provide directory to skip data split.",
    )
    parser.add_argument(
        "--scene-stats-csv",
        type=str,
        default="/opt/ml/processing/input/stats/scene_stats.csv",
        help="Full path to CSV of scene level statistics.",
    )
    parser.add_argument(
        "--bbox-width",
        default=6,
        type=int,
        help="Bounding box width to fill missing widths in training set.",
    )
    parser.add_argument(
        "--bbox-height",
        default=8,
        type=int,
        help="Bounding box height to fill missing heights in training set.",
    )
    parser.add_argument(
        "--data-channels",
        default=[
            "VH_dB",
            "VV_dB",
            "bathymetry",
        ],
        nargs="+",
        help=f"Data channels to include in image chip. Choices {CHANNELS}",
    )
    # Arguments for chipped scenes dataset
    parser.add_argument(
        "--chip-size", default=2560, type=int, help="Size of image chip/tile."
    )
    parser.add_argument(
        "--empty-chip-overlap",
        default=32,
        type=int,
        help="Overlap (in pixels) between adjacent image chips.",
    )
    parser.add_argument(
        "--prop-keep-empty",
        default=0.2,
        type=float,
        help="Proportion of empty chips to include in training set. Default 0.1",
    )
    parser.add_argument(
        "--shoreline-dir",
        type=str,
        default="/opt/ml/processing/input/shoreline/trainval/",
        help="Directory to shoreline data.",
    )

    return parser.parse_args()


def create_xview3_dataset_dict(
    create_xview3_func: Callable,
    gt_labels_csv: Union[str, os.PathLike],
    df_stats: pd.DataFrame,
    output_dir: Union[str, os.PathLike],
    data_channels: List[str],
    bbox_width: int = 6,
    bbox_height: int = 8,
    name_prefix: str = "xview3",
    save_json: bool = False,
) -> Tuple[Union[str, os.PathLike], List[dict]]:
    """Create Detectron2 dataset dict for all scenes.
    Runs `create_xview3_full_scene_annotations` or `create_xview3_chipped_scene_annotations` in a loop for all scenes.

    Args:
        create_xview3_func (Callable): callable function, either `create_xview3_full_scene_annotations`
                or `create_xview3_chipped_scene_annotations`
        gt_labels_csv (Union[str, os.PathLike]): groundtruth CSV of labels.
        df_stats (pd.DataFrame): dataframe of scene image statistics.
        data_channels (List[str]): list of channels to include in dataset.
        bbox_width (int): bounding box width to assign to detections if none present in annotations. default 6.
        bbox_height (int): bounding box height to assign to detections if none present in annotations. default 6.
        output_dir (Union[str, os.PathLike]): destination to save dataset.
        name_prefix (str): prefix to add to dataset name upon saving.
        save_json (bool): Save dataset dict as json. Default False, saves as pickle object.

    Returns:
        dataset_path, dataset (Tuple[Union[str, os.PathLike], List[dict]]): path to saved dataset, dataset_dict
    """
    df_gt = pd.read_csv(gt_labels_csv)

    output_records = []
    scenes = df_gt.scene_id.unique()
    logger.info(f"Creating xView3 detectron2 dataset dict for {len(scenes)} scenes.")

    for scene_id in scenes:
        df_scene_gt = df_gt[df_gt.scene_id == scene_id].reset_index(drop=True)
        df_scene_stats = df_stats[df_stats.scene_id == scene_id].reset_index(drop=True)

        records = create_xview3_func(
            scene_id=scene_id,
            df_scene_gt=df_scene_gt,
            df_scene_stats=df_scene_stats,
            channels=data_channels,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
        )
        output_records.append(records)

    # flatten nested list of records.
    if isinstance(output_records[0], list):
        output_records = list(itertools.chain.from_iterable(output_records))

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    output_fn = Path(output_dir) / f"{name_prefix}-{Path(gt_labels_csv).stem}.dataset"
    logger.info(
        f"#scenes: {len(scenes) }, #images(items) in dataset: {len(output_records)}"
    )

    save_dataset(output_records, output_fn, save_as_json=save_json)

    logger.info(f"Saved dataset_dict to {str(output_fn)}")

    return output_fn, output_records


if __name__ == "__main__":
    start = perf_counter()
    configure_logging()

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    labels_dir = args.gt_labels_dir

    if labels_dir is None:
        assert (
            Path(args.train_labels_csv).is_file()
            and Path(args.valid_labels_csv).is_file()
        ), f"Verify {args.train_labels_csv} and {args.valid_labels_csv} files exist."

        logger.info(
            f"Merging {args.train_labels_csv} and {args.valid_labels_csv} creating new dataset split."
        )
        labels_dir = output_dir / "labels"

        create_data_split(
            train_labels_csv=args.train_labels_csv,
            valid_labels_csv=args.valid_labels_csv,
            tiny_labels_csv=args.tiny_labels_csv,
            output_dir=labels_dir,
            prop_valid_scenes=args.prop_valid_scenes,
            exclude_bay_of_biscay=args.exclude_bay_of_biscay,
            seed=ast.literal_eval(args.seed),
        )

    if args.dataset_type == "full":
        create_dataset_func = create_xview3_full_scene_annotations
        prefix = "xview3-full"

    elif args.dataset_type == "chipped":
        prefix = f"xview3-chipped_{args.chip_size}x{args.chip_size}"

        create_dataset_func = partial(
            create_xview3_chipped_scene_annotations,
            chip_size=args.chip_size,
            empty_chip_overlap=args.empty_chip_overlap,
            prop_keep_empty=args.prop_keep_empty,
            shoreline_dir=args.shoreline_dir,
        )
    else:
        raise ValueError(
            f"Unknown `dataset_type={args.dataset_type}. Choices: `full`, `chipped`"
        )

    d2_datasets_output_dir = output_dir / "detectron2_dataset"
    df_stats = pd.read_csv(args.scene_stats_csv)
    
    for csv in Path(labels_dir).rglob("*.csv"):
        
        create_xview3_dataset_dict(
            create_xview3_func=create_dataset_func,
            gt_labels_csv=csv,
            df_stats=df_stats,
            output_dir=d2_datasets_output_dir,
            data_channels=args.data_channels,
            name_prefix=prefix,
        )

    end = perf_counter()
    elapsed_time = end - start
    logger.info(f"Elasped time: {elapsed_time:.4f} seconds.")
