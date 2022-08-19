"""Save xView3 scene imagery from tarball to file storage (hdf5 or Zarr)

Example usage:

    python store_xview3_imagery.py --input-dir /data/imagery/ --output-dir data/imagery/ --data-split tiny --store-format hdf5
    
"""
import logging
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

from xview3_d2.utils import configure_logging, extract_tarball, save_hdf5, save_zarr

SAVE_FORMAT_FUNC = {"hdf5": save_hdf5, "zarr": save_zarr}
CHANNELS = [
    "VH_dB",
    "VV_dB",
    "bathymetry",
    "owiWindDirection",
    "owiMask",
    "owiWindQuality",
    "owiWindSpeed",
]


def parse_args():
    """Parse commandline arguments."""
    parser = ArgumentParser(description="Save xView3 imagery to File Storage format.")

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/opt/ml/processing/input/",
        help="xView3 data directory. The data can be compressed tar.gz or folders containing individual .tif files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/processing/output/imagery/",
        help="Destination directory.",
    )
    parser.add_argument(
        "--data-split",
        default=None,
        type=str,
        help="Data split name. This will be used as a subfoler in the output dir.",
    )
    parser.add_argument(
        "--store-format",
        default="hdf5",
        type=str,
        choices=["hdf5", "zarr"],
        help="Format to save data.",
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
    parser.add_argument(
        "--resize-ancillary",
        default=True,
        help="Boolean flag to indicate whether to resize ancillary images to match SAR scene size.",
    )
    parser.add_argument(
        "--chunk_size", default=4096, type=int, help="Chunk size for storing data."
    )
    parser.add_argument(
        "--delete-tar",
        default=True,
        help="Boolean flag to indicate whether to delete tar files after extraction.",
    )
    parser.add_argument(
        "--save-zarr-zipstore",
        action="store_true",
        help="Boolean flag to indicate whether to save zarr format with ZipStore instead of DirectoryStore.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def main():
    """Main function to run when script is called. Saves xView3 data to one of two formats: Zarr or Hdf5."""
    start = perf_counter()
    args = parse_args()
    configure_logging(args.verbose)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    store_format = args.store_format
    data_split = args.data_split
    save_format_func = SAVE_FORMAT_FUNC[args.store_format]
    extract_destination = output_dir / "tif"

    if store_format == "zarr" and args.save_zarr_zipstore:
        from functools import partial

        logging.info("Using Zarr ZipStore.")
        save_format_func = partial(save_format_func, zip_store=args.save_zarr_zipstore)
        store_format = "zarr_zip"

    logging.info("Extracting xView3 scenes")

    for scene_file in input_dir.rglob("*.tar.gz"):
        tarball_parent = scene_file.parent
        scene_id = scene_file.name.split(".tar.gz")[0]

        # infer data split from path.
        scene_data_split = (
            str(tarball_parent.relative_to(input_dir))
            if tarball_parent.resolve() != input_dir.resolve()
            else data_split
        )

        extract_destination = output_dir / "tif" / scene_data_split

        extract_tarball(
            scene_file,
            destination=extract_destination,
            delete_tar=args.delete_tar,
        )

        save_format_func(
            scene_dir=extract_destination / scene_id,
            destination_dir=output_dir / store_format / scene_data_split,
            channels=args.data_channels,
            resize_ancillary=args.resize_ancillary,
            chunk_size=args.chunk_size,
        )

    end = perf_counter()
    elapsed_time = end - start
    logging.info(f"Elasped time: {elapsed_time:.4f} seconds.")


if __name__ == "__main__":
    main()
