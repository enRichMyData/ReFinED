from my_tests.utility.test_utils import load_model, run_refined_batch
import pandas as pd
import glob
import time


def eval_2T(
        model: str,
        eval_set: str = "2TR4",
        batch_size: int = 512,
        prediction_mode: str = "cell",
        all_metrics: bool = True,
        verbose: bool = True
):
    # path to data and truth labels
    targets_file = f"my_tests/data/EL_challenge/{eval_set}/targets/CEA_2T_WD_Targets.csv"
    gt_file = f"my_tests/data/EL_challenge/{eval_set}/gt/cea.csv"
    tables_folder = f"my_tests/data/EL_challenge/{eval_set}/tables"

    # load targets and truth labels
    targets_df = pd.read_csv(targets_file, header=None, names=["table", "row", "col"])
    gt_df = pd.read_csv(gt_file, header=None, names=["table", "row", "col", "qid"])

    # targets_df[["row", "col"]] = targets_df[["row", "col"]].astype(int)
    # gt_df[["row", "col"]] = gt_df[["row", "col"]].astype(int)

    eval_df = targets_df.merge(gt_df, on=["table", "row", "col"], how="left") # merged both

    # collect texts and truths
    texts, truths = [], []

    for table_file in glob.glob(f"{tables_folder}/*.csv"):
        table_name = table_file.split("/")[-1].split(".")[0]
        df = pd.read_csv(table_file)


        # select only relevant cells
        table_eval = eval_df[eval_df["table"] == table_name]

        for _, row in table_eval.iterrows():
            row_idx = int(row["row"]) - 1
            col_idx = int(row["col"])

            cell_text = str(df.iat[row_idx, col_idx])
            context_cells = [str(df.iat[row_idx, c]) for c in range(df.shape[1]) if c != col_idx]

            # cell-level prediction
            if prediction_mode == "cell":
                text = str(df.iat[row_idx, col_idx])

            # row-level prediction
            elif prediction_mode == "row":
                text = f"{cell_text},{','.join(context_cells)}"

            # multiple gold entities:
            qids = row["qid"].split()
            for main_qid in qids:
                #TODO endre text slik at main entity er idx[0] - slik at measure_accuracy fungerer
                # eller endre measure_accuarcy til å velge mellom "cell-level" eller "row-level" accuracy
                texts.append(text)
                truths.append([main_qid])

    print("\nTEXTS")
    print(*texts[:10], sep="\n")
    print("\nTRUTHS")
    print(*truths[:10], sep="\n")
    print("\n")


    return None, None, None


if __name__ == "__main__":

    model = "wikipedia_model_with_numbers"

    # refined_model = load_model(device="gpu"", entity_set="wikidata", model=model)
    refined_model = None

    all_spans, truths, duration = eval_2T(
        model=refined_model,
        eval_set="2TR4",
        batch_size=512,
        prediction_mode="cell"
    )
