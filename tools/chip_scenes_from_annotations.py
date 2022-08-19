"""Chip Scenes.

The code below chips large SAR images and ancillary images into desired chip sizes.
It also create COCO-style annotations for each image chip and saves the annotations
in the parent directory of the image chips for each scene.
"""
import argparse
import itertools
import os
import tarfile
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import List, Union

import h5py
import numpy as np
import rasterio
import zarr
from rasterio.enums import Resampling
from xview3_d2.data.datasets.xview3 import CHANNELS
from xview3_d2.utils import configure_logging, open_dataset, timer

logger = configure_logging("Chip-Scenes", verbose=False)


def parse_arguments():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description="Create image chips from scene.")
    parser.add_argument(
        "--scenes-input-dir",
        type=str,
        default="/opt/ml/processing/input/scenes/",
        help="Path to folder containing files.",
    )
    parser.add_argument(
        "--d2-dataset",
        type=str,
        default="/opt/ml/processing/input/dataset",
        help="Detectron2 dataset dict by `create_chipped_scenes_annotations.py`",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/processing/output/",
        help="Output folder to store chipped images.",
    )
    parser.add_argument(
        "--channels",
        default=[
            "VH_dB",
            "VV_dB",
            "bathymetry",
        ],
        nargs="+",
        help=f"Data channels to include in image chip. Choices {CHANNELS}",
    )
    parser.add_argument(
        "--save-fmt",
        type=str,
        default="npz",
        help="Storage format to save image chips.",
        choices=["zarr", "hdf5", "npz"],
    )
    args = parser.parse_args()
    return args


def read_scene_files(
    scene_tar_file: Union[str, os.PathLike],
    scene_height: int,
    scene_width: int,
    channels: List[str] = ["VH_dB", "VV_dB"],
):
    """
    Read scene files with rasterio and store in numpy memmap.

    Args:
        scene_tar_file (Union[str, os.PathLike]): filename of compressed scene file.
        scene_height (int):                       scene height of SAR channel
        scene_width (int):                        scene width of SAR image channel
        channels (List[str]], optional):          list of data channels to include in image stack.
                                                  Defaults to ['VH_dB', 'VV_dB'].

    Returns:
        scene_memmap: scene memmap array
        channel_map (dict): mapping memmap channel index to channel name
        tmp_fname (os.PathLike): temporary file name of memmap.
    """

    with TemporaryDirectory() as tmp_path:
        with tarfile.open(name=str(scene_tar_file), mode="r:gz") as tar:
            tar.extractall(tmp_path)

        files = [f for f in Path(tmp_path).rglob("*.tif") if f.stem in channels]

        scene_store = {}
        for f in files:
            # Read each channel and if necessary upsample to SAR scene size.
            with rasterio.open(f) as src:

                img = src.read(
                    1,
                    out_shape=(scene_height, scene_width),
                    resampling=Resampling.bilinear,
                )
            scene_store[f.stem] = img

    return scene_store


@timer
def chip_scenes(
    records: List,
    scene_tar_file: Union[str, os.PathLike],
    channels: List,
    destination_dir: Union[str, os.PathLike],
    save_fmt: str,
):
    """Chip scenes.

    Args:
        records (List): detectron2 dataset dict list of annotations
        scene_tar_file (Union[str, os.PathLike]): filename of compressed scene tarball
        channels (List): list of channels to include.
        destination_dir (Union[str, os.PathLike]): directory to save chipped outputs.
        save_fmt (str): file format to store image chips.
    """
    logger.info(f"Scene ID: {records[0]['scene'].upper()} | Loaded images.")
    scene_height = records[0]["scene_height"]
    scene_width = records[0]["scene_width"]

    # Read scene files.
    scene_store = read_scene_files(scene_tar_file, scene_height, scene_width, channels)

    for counter, record in enumerate(records):
        chip_top = record["scene_chip_top"]
        chip_left = record["scene_chip_left"]

        img_chips = {
            chan: scene[
                chip_top : chip_top + record["height"],
                chip_left : chip_left + record["width"],
            ]
            for chan, scene in scene_store.items()
        }
        save_chip_fn = destination_dir / record["file_name"]

        if save_fmt == "zarr":
            zarr.convenience.save_group(save_chip_fn.with_suffix('.zarr'), **img_chips)
        elif save_fmt == "npz":
            np.savez(save_chip_fn.with_suffix('.npz'), **img_chips)
        elif save_fmt == "hdf5":
            with h5py.File(str(save_chip_fn.with_suffix('.h5')), mode='w') as h5:
                for ch, img in img_chips.items():
                    h5.create_dataset(ch, data=img, compression="gzip", shuffle=True)

    logger.debug(
        f"Scene ID: {record['scene'].upper()} | Saved {counter + 1} chips saved to: {destination_dir}."
    )


def main():
    """Run main function to chip image scenes."""
    start = perf_counter()
    args = parse_arguments()
    channels = set(args.channels)
    dataset_fname = Path(args.d2_dataset)
    output_dir = Path(args.output_dir) / dataset_fname.stem.replace("-", "_")
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    dataset = open_dataset(dataset_fname)

    if isinstance(dataset[0], list):
        # flatten nested list.
        dataset = list(itertools.chain.from_iterable(dataset))
    scene_dataset = defaultdict(list)

    # group by scene.
    for ds in dataset:
        scene_dataset[ds["scene"]].append(ds)

    # loop through scenes in dataset and chip scenes.
    for scene_id, records in scene_dataset.items():
        scene_tar_file = Path(args.scenes_input_dir) / f"{scene_id}.tar.gz"

        if not scene_tar_file.is_file():
            logger.debug(f"{scene_tar_file} does not exist.")
            continue

        chip_scenes(
            records=records,
            scene_tar_file=scene_tar_file,
            channels=channels,
            destination_dir=output_dir,
            save_fmt=args.save_fmt,
        )

    end = perf_counter()
    elapsed_time = end - start
    logger.info(f"Elasped time: {elapsed_time:.4f} seconds.")


if __name__ == "__main__":
    main()
