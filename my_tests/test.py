from refined.dataset_reading.entity_linking.dataset_factory import Datasets
from my_tests.utility.test_utils import load_model
from my_tests.accuracy import evaluate_refined, measure_accuracy
from refined.data_types.doc_types import Doc
from refined.data_types.base_types import Entity, Span
from typing import Iterable


# refined_model = load_model(device="cpu", model="wikipedia_model_with_numbers", entity_set="wikidata")

# ------- Run official evaluation -------
# metrics = evaluate_refined(refined_model, "HTR1")


#TODO: Bruk dette for refined-evaluering av HTR data !!!
#TODO: sett inn under "dataset_factory.py" under andre docs

# hardtable docs testing                 
class Test:
    def __init__(self):
        self.preprocessor = None

    def get_hardtable_docs(
            self,
            split: str,
            include_spans: bool = True,
            include_gold_label: bool = True,
            filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        import json
        import glob
        import pandas as pd

        # Paths
        tabl_folder = f"my_tests/data/EL_challenge/{split}/tables"
        cell_to_qid_file = f"my_tests/data/EL_challenge/{split}/cell_to_qid.json"

        with open(cell_to_qid_file, "r") as f:
            cell_to_qid = json.load(f)

        docs = []
        for table_file in glob.glob(f"{tabl_folder}/*.csv"):
            table_name = table_file.split("/")[-1].split(".")[0]
            if table_name not in cell_to_qid:
                continue

            # store table in pandaframe
            df = pd.read_csv(table_file)
            for row_idx in range(df.shape[0]):
                # row text concatenated with commas (row-level context)
                text = ",".join(str(df.iat[row_idx, c]) for c in range(df.shape[1]))

                print(text)

            break

            #     spans = []
            #     for cell_id, qids in cell_to_qid[table_name].items():
            #         r, c = map(int, cell_id.split("-"))
            #         if r != row_idx: 
            #             continue

            #         cell_text = str(df.iat[r, c])
            #         start = text.find(cell_text)
            #         if start == -1: 
            #             continue

            #         gold_entity = None
            #         if include_gold_label and qids:
            #             gold_entity = Entity(
            #                 wikidata_entity_id=qids[0],
            #                 wikipedia_entity_title="",
            #                 human_readable_name=cell_text
            #             )
                    
            #         span = Span(
            #             text=cell_text,
            #             start=start,
            #             ln=len(cell_text),
            #             gold_entity=gold_entity,
            #             coarse_type="MENTION"
            #         )
            #         spans.append(span)

            #     doc = Doc.from_text_with_spans(
            #         text=text,
            #         spans=spans if include_spans else [],
            #         md_spans=None,
            #         preprocessor=self.preprocessor
            #     )
            #     docs.append(doc)

            # return doc

# t = Test()
# t.get_hardtable_docs("HTR1")

def generate_samples(n=10, correct_prob=0.6, no_pred_prob=0.2):
    import random
    pred_spans = []
    truths = []

    for i in range(n):
        truth_qid = f"Q{i+1}"
        truths.append((0, i, [truth_qid]))

        r = random.random()
        if r < no_pred_prob:
            # no prediction
            pred_spans.append([])
        elif r < no_pred_prob + correct_prob:
            # correct prediction
            pred_spans.append([Span(text=f"Cell{i+1}", start=i*6, ln=5, predicted_entity=Entity(wikidata_entity_id=truth_qid))])
        else:
            # wrong prediction
            wrong_qid = f"Q{n + i + 1}"  # ensure it's different from truth
            pred_spans.append([Span(text=f"Cell{i+1}", start=i*6, ln=5, predicted_entity=Entity(wikidata_entity_id=wrong_qid))])

    return pred_spans, truths


pred_spans, truths = generate_samples(50)

accuracy = measure_accuracy(pred_spans=pred_spans, truths=truths, all_metrics=True, verbose=True)