import argparse



MODEL_CHOICES = {
    "wikipedia_model_with_numbers": "wikipedia_model_with_numbers",
    "wikipedia_model": "wikipedia_model",
    "merged_10k": "fine_tuned_models/merged_10k/f1_0.9229",
    "merged_60k": "fine_tuned_models/merged_60k/f1_0.9254",
    "companies_full": "fine_tuned_models/companies_full/f1_0.8711",
    "movies_full": "fine_tuned_models/movies_full/f1_0.9237",
    "merged_full": "fine_tuned_models/merged_full/f1_0.8972"
}

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
        type=lambda s: s.upper(),
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
    parser.add_argument(
        "--entity_set", "-es",
        type=str.lower,
        dest="entity_set",
        choices=["wikidata", "wikipedia"],
        default="wikidata",
        help="Entity set to use (wikidata or wikipedia"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        dest="model",
        choices=list(MODEL_CHOICES.keys()),
        default="wikipedia_model_with_numbers",
        help="Model to use (shrot alias, maps to full path internally)"
    )

    args = parser.parse_args()
    return args