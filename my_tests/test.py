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



# measure accuracy testing
pred_spans = [
    [Span(text="Cell1", start=0, ln=5, predicted_entity=Entity(wikidata_entity_id="Q1"))],   # correct
    [Span(text="Cell2", start=6, ln=5, predicted_entity=Entity(wikidata_entity_id="Q3"))],   # wrong
    [],  # no prediction
    [Span(text="Cell4", start=12, ln=5, predicted_entity=Entity(wikidata_entity_id="Q4"))],  # correct
    [Span(text="Cell5", start=18, ln=5, predicted_entity=Entity(wikidata_entity_id="Q0"))],  # no prediction
    [Span(text="Cell6", start=24, ln=5, predicted_entity=Entity(wikidata_entity_id="Q6"))],  # correct
    [Span(text="Cell7", start=30, ln=5, predicted_entity=Entity(wikidata_entity_id="Q7"))],  # correct
    [],  # no prediction
    [Span(text="Cell9", start=36, ln=5, predicted_entity=Entity(wikidata_entity_id="Q9"))],  # correct
    [Span(text="Cell10", start=42, ln=5, predicted_entity=Entity(wikidata_entity_id="Q10"))],# correct
]

truths = [
    (0, 0, ["Q1"]),
    (0, 1, ["Q2"]),
    (0, 2, ["Q3"]),
    (0, 3, ["Q4"]),
    (0, 4, ["Q5"]),
    (0, 5, ["Q6"]),
    (0, 6, ["Q7"]),
    (0, 7, ["Q8"]),
    (0, 8, ["Q9"]),
    (0, 9, ["Q10"]),
]

accuracy = measure_accuracy(pred_spans=pred_spans, truths=truths, all_metrics=True, verbose=True)