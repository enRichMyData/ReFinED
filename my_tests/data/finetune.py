import os
import pandas as pd

from typing import Iterable

from refined.data_types.doc_types import Doc, Span

data_path = "data"
filename = "movies_train.csv"

folder = filename.split("_")[0]
train_text = os.path.join(data_path, folder, filename)
train_truth = os.path.join(data_path, folder, f"el_{folder}_gt_wikidata.csv")

# Load the text and truth data
dftext = pd.read_csv(train_text)
dftruth = pd.read_csv(train_truth)
dftruth = dftruth[dftruth["tableName"] == f"{folder}_train"]

# Merge the two DataFrames on the idRow column
frame = pd.merge(dftext, dftruth, left_index=True, right_on="idRow")

print(frame.iloc[0])

def get_wikidata_docs(frame) -> Iterable[Doc]:
    docs = []
    for _, row in frame.iterrows():
        text = f"{row['Title']},{row['Release_Date']},{row['Duration']},{row['Genre']},{row['Country']},{row['Language']},{row['Director']}"

        gold_entity = row['entity']

        start = text.find(row['Title'])
        lenght = len(row['Title'])

        spans = [
            Span(
                text=row['Title'],
                start=start,
                ln=lenght,
                gold_entity=gold_entity,
                coarse_type="MENTION"
            )
        ]

        # --- Mention detection spans (other useful info) ---
        md_spans = [
            Span(
                text=row['Title'],
                start=start,
                ln=lenght,
                gold_entity=None,
                coarse_type="MENTION"
            )
        ]

        # Relase data as DATE
        if pd.notna(row["Release_Date"]):
            date_start = text.find(row['Release_Date'])
            if date_start != -1:
                md_spans.append(
                    Span(
                        text=str(row['Release_Date']),
                        start=date_start,
                        ln=len(str(row['Release_Date'])),
                        gold_entity=None,
                        coarse_type="DATE"
                    )
                )

        # Duration as QUANTITY
        if pd.notna(row["Duration"]):
            dur_start = text.find(str(row["Duration"]))
            if dur_start != -1:
                md_spans.append(
                    Span(
                        text=str(row["Duration"]),
                        start=dur_start,
                        ln=len(str(row["Duration"])),
                        gold_entity=None,
                        coarse_type="QUANTITY",
                    )
                )

        # for Country, Language, Director as generic mentions
        for col in ["Country", "Language", "Director"]:
            if pd.notna(row[col]):
                val = str(row[col])
                val_start = text.find(val)
                if val_start != -1:
                    md_spans.append(
                        Span(
                            text=val,
                            start=val_start,
                            ln=len(val),
                            gold_entity=None,
                            coarse_type="MENTION",
                        )
                    )

        doc = Doc.from_text_with_spans(
            text=text,
            spans=spans,
            md_spans=md_spans,
            preprocessor=self.preprocessor
        )

        docs.append(doc)

        print(f"{'Text:':<15}{text}")
        print(f"{'Gold Entity:':<15}{gold_entity}")
        break

    return docs

get_wikidata_docs(frame)