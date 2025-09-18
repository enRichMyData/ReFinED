from my_tests.setup import *

import os
import time
import sys
import torch
import pandas as pd

def main():

    # ======== CONFIG === ======== #
    USE_CPU = False
    NO_LINES = 1
    DEFAULT_DATA_FOLDER = "my_tests/data"
    # ============================ #

    # Handles command line arguments
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input_file>")
        print("Supported files:\n- 'imdb_top_100.csv'\n- 'companies_test.csv'\n- 'movies_test.csv'\n- 'SN_test.csv'")
        sys.exit(1)

    input_file = sys.argv[1]
    try: texts = load_input_file(input_file, DEFAULT_DATA_FOLDER)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Load model
    refined_model = load_model(USE_CPU=USE_CPU)

    # Retrieve texts from running ReFinED entity3. linker
    texts = texts[:NO_LINES]

    # runs entity linking, times it
    start_time = time.time()
    all_spans = run_refined(texts=texts, model=refined_model)
    duration = time.time() - start_time



    # go through each of the lines from csv
    print("\n" + "-" * 60 + "\n")
    for raw_line, doc_spans in zip(texts, all_spans):

        print(f"[{raw_line}]")  # line from CSV
        for span in doc_spans[:1]:
            pred_ent = span.predicted_entity
            pred_qid = getattr(pred_ent, "wikidata_entity_id", None)        # gets QID
            pred_title = getattr(pred_ent, "wikipedia_entity_title", None)  # gets title, aka text

            print(f"=== [ {pred_title} ] [ {pred_qid} ] ===")
        print("\n" + "-"*60 + "\n")

    print(f"\nInference time for {len(texts)} texts: {duration:.2f} seconds")


    # ============ CPU SWITCH ======================= #
    print("CUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available() and not USE_CPU:
        print("Running on GPU:", torch.cuda.get_device_name(0) + "\n")
    else:
        print("Running on CPU\n")
    # =============================================== #

    df = pd.read_json("my_tests/data/companies_mention_to_qids.json")
    qid = df["WeoGeo"][0]
    print("Devil fruit (WeoGeo): " + qid)


if __name__ == "__main__":
    main()