"""Utility functions."""

import json
import logging
import os
import pickle
import tarfile
from datetime import datetime
from functools import partial, wraps
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import rasterio
import zarr
from rasterio.enums import Resampling
from rasterio.windows import Window

# from constants import SAR_CHANNELS

__all__ = [
    "configure_logging",
    "extract_files",
    "save_hdf5",
    "save_zarr",
    "load_image_rasterio",
    "load_image_zarr",
    "load_image_hdf5",
    "get_seed_value",
    "open_dataset",
    "timer",
    "cli_argument",
    "save_dataset",
    "subcommand",
]

# SAR channels
SAR_CHANNELS = ["VH_dB", "VV_dB"]


def get_seed_value() -> int:
    """Get seed value.
    Adapted from https://github.com/facebookresearch/detectron2/blob/224cd2318fdb45b5e22bbb861ee9711ee52c8b75/detectron2/utils/env.py#L35
    """
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed

def get_rng(seed:Optional[int]):
    if seed is None:
        seed = get_seed_value()
    return np.random.default_rng(seed) 

class Encoder(json.JSONEncoder):
    """Json encoder."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_dataset(
    data: Iterable, fname: Union[str, os.PathLike], save_as_json: bool = False
):
    """Save dataset annotations.

    Args:
        data (Iterable): data to be saved.
        fname (Union[str, os.PathLike]): filename to save.
        save_as_json (bool): save file as json.
    """
    fname = Path(fname)

    if save_as_json:
        output_fn = output_fn.with_suffix(".json")
        with output_fn.open("w") as f:
            json.dump(data, f, cls=Encoder)
    else:
        with fname.open("wb") as f:
            pickle.dump(data, f)


def open_dataset(fname: Union[str, os.PathLike]):
    """Open dataset annotations.

    Args:
        fname (Union[str, os.PathLike]): name of file to open.
    """
    fname = Path(fname)

    if "json" in fname.suffix:
        with fname.open("r") as f:
            return json.load(f)

    with fname.open("rb") as f:
        data = pickle.load(f)
    return data


def configure_logging(name=None, verbose: bool = True):
    """Setup logging."""
    logging_level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger(__name__ if name is None else name)
    logging.basicConfig(level=logging_level)
    root_logger.setLevel(logging_level)

    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter("%(levelname)-8s %(message)s")
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    return root_logger


def timer(func: Callable, logger=None) -> Callable:
    """Timer decorator

    Args:
        func (Callable): _description_
        logger (_type_): _description_

    Returns:
        Callable: input function
    """
    if logger is None:
        logger = logging.getLogger(func.__name__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        output = func(*args, **kwargs)
        end = perf_counter()
        elapsed_time = end - start
        logger.debug(f"{func.__name__} Elasped time: {elapsed_time:.4f} seconds.")
        return output

    return wrapper


def extract_tarball(
    file: Union[os.PathLike, str],
    destination: Union[os.PathLike, str],
    delete_tar: bool,
):
    """Extract single tarball.

    Args:
        file (Union[os.PathLike, str]): compresseds file to extract.
        destination (Union[os.PathLike, str]): destination to save extracted files.
        delete_tar (bool): delete tar file after extraction. Defaults to False.
    """
    with tarfile.open(name=str(file), mode="r:gz") as tar:
        tar.extractall(destination)

    if delete_tar:
        os.remove(file)


@timer
def extract_files(
    files: Iterable[Union[os.PathLike, str]],
    destination_dir: Union[os.PathLike, str],
    delete_tar: bool = False,
    num_cpus: Optional[int] = None,
):
    """Extract compressed tar.gz files using multiprocessing.

    Args:
        files (Iterable[os.PathLike, str]): List of full paths to compressed tar.gz files.
        destination (Union[os.PathLike, str]): destination to save extracted files.
        delete_tar (bool, optional): delete tar file after extraction. Defaults to False.
        num_cpus (_type_, optional): Number of CPUs to use to extract. Defaults to None, uses maximum number of CPUs.
    """
    logger = logging.getLogger("ExtractImages")
    num_processes = min(cpu_count(), len(files)) if num_cpus is None else num_cpus
    extract_tar_worker_func = partial(
        extract_tarball, destination=destination_dir, delete_tar=delete_tar
    )

    logger.info(f"extracting {len(files)} files with {num_processes} CPU processes")

    with Pool(
        processes=num_processes,
    ) as pool:
        pool.map(extract_tar_worker_func, files)


def parse_chunks(chunk_size):
    if isinstance(chunk_size, int):
        return (chunk_size, chunk_size)
    elif isinstance(chunk_size, tuple):
        return chunk_size
    else:
        return True


@timer
def save_hdf5(
    scene_dir: Union[str, os.PathLike],
    destination_dir: Union[str, os.PathLike],
    channels: Union[Tuple, List],
    resize_ancillary: bool = False,
    use_cv2: bool = True,
    chunk_size: Optional[Union[int, Tuple[int, int]]] = None,
):
    """Save imagery data to hdf5.

    Args:
        scene_dir (Union[str, os.PathLike]): Scene directory containing .tif files.
        destination_dir (Union[str, os.PathLike]): Directory to save .h5 data
        channels (Union[Tuple, List]): Scene data channels to include.
        resize_ancillary (bool): Boolean indicator to resize ancillary (non-SAR channels). Default False,
        use_cv2 (bool): Boolean indicator to use OpenCV for resize. Default True.
        chunk_size (int, tuple(int, int)): Chunk size to store data.
    """

    scene_id = Path(scene_dir).name
    Path(destination_dir).mkdir(exist_ok=True, parents=True)
    h5_fname = Path(destination_dir) / f"{scene_id}.h5"
    channels = set(channels)
    chan_fnames = {f.stem: f for f in scene_dir.glob("*.tif") if f.stem in channels}
    sar_channels = tuple(set(chan_fnames).intersection(set(SAR_CHANNELS)))
    ancillary_channels = tuple(set(chan_fnames).symmetric_difference(set(SAR_CHANNELS)))
    ordered_channels = sar_channels + ancillary_channels
    scene_height = None
    scene_width = None
    chunks = parse_chunks(chunk_size)

    with h5py.File(str(h5_fname), "w") as h5_file:
        for chan in ordered_channels:
            img = load_image_rasterio(
                chan_fnames[chan],
                channel=chan,
                scene_height=scene_height,
                scene_width=scene_width,
                use_cv2=use_cv2,
            )
            h5_file.create_dataset(
                chan,
                data=img,
                compression="gzip",
                shuffle=True,
                chunks=chunks,
            )

            if resize_ancillary and chan in sar_channels:
                scene_height, scene_width = img.shape


@timer
def save_zarr(
    scene_dir: Union[str, os.PathLike],
    destination_dir: Union[str, os.PathLike],
    channels: Union[Tuple, List],
    zip_store: bool = False,
    chunk_size: Optional[Union[int, Tuple[int, int]]] = None,
):
    """Save imagery data to zarr.

    Args:
        scene_dir (Union[str, os.PathLike]): Scene directory containing .tif files.
        destination_dir (Union[str, os.PathLike]): Directory to save .h5 data
        channels (Union[Tuple, List]): Scene data channels to include.
        zip_store (bool): Store zarr data in zip. Default False, stores in DirectoryStore.
        chunk_size (int, tuple(int, int)): Chunk size to store data.
    """
    scene_id = Path(scene_dir).name
    Path(destination_dir).mkdir(exist_ok=True, parents=True)
    zarr_fname = Path(destination_dir) / f"{scene_id}.zarr"
    channels = [chan.lower() for chan in channels]
    chunks = parse_chunks(chunk_size)

    if zip_store:
        store = zarr.ZipStore(zarr_fname, mode="w")
    else:
        store = zarr.DirectoryStore(zarr_fname)
    zarr_group_data = zarr.group(store=store, overwrite=True)

    for f in scene_dir.glob("*.tif"):
        if f.stem.lower() in channels:
            with rasterio.open(str(f)) as src:
                img = src.read(
                    1,
                )
            zarr_group_data.create_dataset(f.stem, data=img, chunks=chunks)
    if zip_store:  # need to add this for ZipStore
        store.close()


@timer
def load_image_rasterio(
    fname,
    channel: str,
    scene_width: Optional[int] = None,
    scene_height: Optional[int] = None,
    coords: Optional[Tuple[int, int, int, int]] = None,
    use_cv2: bool = True,
) -> np.ndarray:
    """Load image with rasterio.

    Args:
        fname (str): path to scene id to open.
        scene_width (int): maximum width of SAR scene. Used to resize ancillary images.
                           If None provided, no resize.
        scene_height (int): maximum height of SAR scene. Used to resize ancillary images.
                            If None provided, no resize.
        coords (tuple): coordinate to crop, if provided. In order x0, y0, x1, y1.
        use_cv2 (bool): Use cv2 to resize image

    Returns:
        data (np.ndarray): image.
    """
    crop = False
    window = None

    if coords:
        x0, y0, x1, y1 = coords
        window = Window.from_slices((y0, y1), (x0, x1))
        crop = True

    with rasterio.open(fname) as src:
        if channel in SAR_CHANNELS:
            data = src.read(1, window=window)

        else:
            if use_cv2:  # Faster than rasterio and yields similar results.
                data = cv2.resize(
                    src.read(
                        1,
                    ),
                    (scene_width, scene_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                data = src.read(
                    1,
                    resampling=Resampling.bilinear,
                    out_shape=(scene_height, scene_width),
                )

            if crop:
                data = data[y0:y1, x0:x1]

    return data


@timer
def load_image_hdf5(
    fname,
    channel: str,
    scene_width: Optional[int] = None,
    scene_height: Optional[int] = None,
    coords: Optional[Tuple[int, int, int, int]] = None,
) -> dict:
    """Load image from hdf5 or zarr file store.

    Args:
        fname (str): path to scene id to open.
        channel (str): channel name.
        coords (tuple): coordinate to crop in order x0, y0, x1, y1. If None provided, full image returned.

    Returns:
        data (np.ndarray): image.
    """

    with h5py.File(fname) as src:
        assert channel in list(
            src.keys()
        ), f"Channel: {channel} not in {list(src.keys())}"
        data = src[channel]

        if coords is None:
            return data[:]
        else:
            x0, y0, x1, y1 = coords
            return data[y0:y1, x0:x1]


@timer
def load_image_zarr(
    fname,
    channel: str,
    scene_width: Optional[int],
    scene_height: Optional[int],
    coords: Optional[Tuple[int, int, int, int]],
) -> dict:
    """Load image from zarr file store.

    Args:
        fname (str): path to scene id to open.
        channel (str): channel name.
        scene_width (int): maximum width of SAR scene. Used to resize ancillary images.
        scene_height (int): maximum height of SAR scene. Used to resize ancillary images.
        coords (tuple): coordinate to crop in order x0, y0, x1, y1. If None provided, full image returned.

    Returns:
        data (np.ndarray): image.
    """
    resize = None not in [scene_height, scene_width]

    with zarr.open(fname) as src:
        assert channel in list(
            src.array_keys()
        ), f"Channel: {channel} not in {list(src.array_keys())}"
        data = src[channel]

    if coords is None:
        y1, x1 = data.shape
        coords = (0, 0, x1, y1)

    if resize:
        data = cv2.resize(
            data,
            (scene_width, scene_height),
            interpolation=cv2.INTER_LINEAR,
        )

    if coords:
        x0, y0, x1, y1 = coords
        data = data[y0:y1, x0:x1]

    return data


def cli_argument(*name_or_flags, **kwargs):
    """Convenience function to properly format arguments to pass to the
    subcommand decorator.

    Adapted from https://gist.github.com/mivade/384c2c41c3a29c637cb6c603d4197f9f

    """

    return (list(name_or_flags), kwargs)


def subcommand(args, parent):
    """Decorator to define a new subcommand in a sanity-preserving way.
    The function will be stored in the ``func`` variable when the parser
    parses arguments so that it can be called directly like so::
        args = cli.parse_args()
        args.func(args)
    Usage example::
        @subcommand([argument("-d", help="Enable debug mode", action="store_true")])
        def subcommand(args):
            print(args)
    Then on the command line::
        $ python cli.py subcommand -d

    Adapted from: https://gist.github.com/mivade/384c2c41c3a29c637cb6c603d4197f9f
    """

    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)

    return decorator
