import argparse


def parse_args(supported_files=None):
    """
    Parses command line arguments for input CSV and optional --verbose.
    supported_files: list of filenames to display in help.
    """
    if supported_files is None:
        supported_files = ['imdb_top_100.csv', 'companies_test.csv', 'movies_test.csv', 'SN_test.csv']

    parser = argparse.ArgumentParser(
        description="Run ReFinED entity linking and measure accuracy.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed prediction info"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input CSV file. Supported:\n" +
             "\n".join([f"\t-'{f}'" for f in supported_files])
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device to run on (default: gpu)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["CSV", "JSON"],
        default="CSV",
        help="Format of the ground truth file (default: CSV)"
    )
    parser.add_argument(
        "--no-batch",
        action="store_false",
        dest="batch",
        help="Disable batching (default is batching enabled)"
    )

    args = parser.parse_args()
    return args