"""Merge challenge train and validation splits and create new split."""

from argparse import ArgumentParser
from pathlib import Path

from xview3_d2.data.datasets.xview3 import CHANNELS, create_data_split
from xview3_d2.utils import configure_logging

logger = configure_logging("Split-Train-Val")


def parse_args():
    """Parse commandline arguments."""
    parser = ArgumentParser(description="Save xView3 imagery to File Storage format.")

    parser.add_argument(
        "--train-labels-csv", "trn", required=True, help="Train labels csv."
    )
    parser.add_argument(
        "--valid-labels-csv", "val", required=True, help="Validation labels csv."
    )
    parser.add_argument(
        "--tiny-labels-csv",
        "tiny",
        required=False,
        default=None,
        help="Tiny subset labels csv.",
    )
    parser.add_argument(
        "--prop-valid-scenes",
        "p",
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
        default="/opt/ml/output/data/labels/",
        help="Directory to save split data.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    create_data_split(
        train_labels_csv=args.train_labels_csv,
        valid_labels_csv=args.valid_labels_csv,
        tiny_labels_csv=args.tiny_labels_csv,
        output_dir=output_dir,
        prop_valid_scenes=args.prop_valid_scenes,
        exclude_bay_of_biscay=args.exclude_bay_of_biscay,
        seed=args.seed,
    )
