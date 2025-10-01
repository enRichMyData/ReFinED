import os
import pandas as pd

from typing import Iterable

from refined.data_types.doc_types import Doc
from refined.data_types.base_types import Entity, Span


def get_movie_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = True,
        filter_not_in_kb: bool = True,
) -> Iterable[Doc]:
    """
    Load movie documents with entity spans for fine-tuning.
    """
    data_path = "data"
    filename = f"movies_{split}.csv"   # extendable beyond train
    folder = filename.split("_")[0]

    # Paths
    text_path = os.path.join(data_path, folder, filename)
    truth_path = os.path.join(data_path, folder, f"el_{folder}_gt_wikidata.csv")

    # Load CSVs
    dftext = pd.read_csv(text_path)
    dftruth = pd.read_csv(truth_path)
    dftruth = dftruth[dftruth["tableName"] == f"{folder}_{split}"]  # filter to relevant split
    dftruth["entity"] = dftruth["entity"].str.replace("http://www.wikidata.org/entity/", "", regex=False) # only use QID

    # Optionally filter out rows not in KB
    if filter_not_in_kb:
        dftruth = dftruth[dftruth["entity"].notna() & (dftruth["entity"] != "Q-1")]

    # Merge by row index
    frame = pd.merge(dftext, dftruth, left_index=True, right_on="idRow")

    docs = []
    for _, row in frame.iterrows():
        # Create document text
        text = f"{row['Title']},{row['Release_Date']},{row['Duration']},{row['Genre']},{row['Country']},{row['Language']},{row['Director']}"

        # Gold entity if available
        gold_entity = None
        if include_gold_label and pd.notna(row["entity"]):
            gold_entity = Entity(
                wikidata_entity_id=row["entity"],
                wikipedia_entity_title=row["Title"].replace(" ", "_"),
                human_readable_name=row["Title"]
            )

        start = text.find(row['Title'])
        length = len(row['Title'])

        spans, md_spans = [], []
        if include_spans:
            # EL span (with gold)
            spans = [
                Span(
                    text=row['Title'],
                    start=start,
                    ln=length,
                    gold_entity=gold_entity,
                    coarse_type="MENTION"
                )
            ]

            # MD span (no gold)
            md_spans = [
                Span(
                    text=row['Title'],
                    start=start,
                    ln=length,
                    gold_entity=None,
                    coarse_type="MENTION"
                )
            ]

            # Release date
            if pd.notna(row["Release_Date"]):
                date_str = str(row["Release_Date"])
                date_start = text.find(date_str)
                if date_start != -1:
                    md_spans.append(
                        Span(
                            text=date_str,
                            start=date_start,
                            ln=len(date_str),
                            gold_entity=None,
                            coarse_type="DATE"
                        )
                    )

            # Duration
            if pd.notna(row["Duration"]):
                dur_str = str(row["Duration"])
                dur_start = text.find(dur_str)
                if dur_start != -1:
                    md_spans.append(
                        Span(
                            text=dur_str,
                            start=dur_start,
                            ln=len(dur_str),
                            gold_entity=None,
                            coarse_type="QUANTITY"
                        )
                    )

            # Country, Language, Director
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
                                coarse_type="MENTION"
                            )
                        )

        # Build doc
        doc = Doc.from_text_with_spans(
            text=text,
            spans=spans,
            md_spans=md_spans,
            preprocessor=self.preprocessor
        )
        docs.append(doc)

    return docs


get_movie_docs(None, "train")

def get_companies_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = True,
        filter_not_in_kb: bool = True,
) -> Iterable[Doc]:
    """
    Load company documents with entity spans for fine-tuning.
    """
    data_path = "data"
    filename = f"companies_{split}.csv"   # extendable beyond train
    folder = filename.split("_")[0]

    # Paths
    text_path = os.path.join(data_path, folder, filename)
    truth_path = os.path.join(data_path, folder, f"el_{folder}_gt_wikidata.csv")

    # Load CSVs
    dftext = pd.read_csv(text_path)
    dftruth = pd.read_csv(truth_path)
    dftruth = dftruth[dftruth["tableName"] == f"{folder}_{split}"]  # filter to relevant split
    dftruth["entity"] = dftruth["entity"].str.replace("http://www.wikidata.org/entity/", "", regex=False) # only use QID

    # Optionally filter out rows not in KB
    if filter_not_in_kb:
        dftruth = dftruth[dftruth["entity"].notna() & (dftruth["entity"] != "Q-1")]

    # Merge by row index
    frame = pd.merge(dftext, dftruth, left_index=True, right_on="idRow")

    docs = []
    for _, row in frame.iterrows():
        # Create document text
        text = f"{row['company']},{row['Founded Year']},{row['Website']},{row['X (Twitter)']},{row['Employees']},{row['Coordinates']},{row['Founded By']},{row['Industry']},{row['Country']},{row['Headquarters']}"

        # Gold entity if available
        gold_entity = None
        if include_gold_label and pd.notna(row["entity"]):
            gold_entity = Entity(
                wikidata_entity_id=row["entity"],
                wikipedia_entity_title=row["company"].replace(" ", "_"),
                human_readable_name=row["company"]
            )

        start = text.find(row['company'])
        length = len(row['company'])

        spans, md_spans = [], []
        if include_spans:
            # EL span (with gold)
            spans = [
                Span(
                    text=row['company'],
                    start=start,
                    ln=length,
                    gold_entity=gold_entity,
                    coarse_type="MENTION"
                )
            ]

            # MD span (no gold)
            md_spans = [
                Span(
                    text=row['company'],
                    start=start,
                    ln=length,
                    gold_entity=None,
                    coarse_type="MENTION"
                )
            ]

            # Release date
            if pd.notna(row["Founded Year"]):
                date_str = str(row["Founded Year"])
                date_start = text.find(date_str)
                if date_start != -1:
                    md_spans.append(
                        Span(
                            text=date_str,
                            start=date_start,
                            ln=len(date_str),
                            gold_entity=None,
                            coarse_type="DATE"
                        )
                    )

            # Country, Language, Director
            for col in ["Website", "X (Twitter)", "Employees", "Coordinates", "Founded By", "Industry", "Country", "Headquarters"]:
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
                                coarse_type="MENTION"
                            )
                        )

        # Build doc
        doc = Doc.from_text_with_spans(
            text=text,
            spans=spans,
            md_spans=md_spans,
            preprocessor=self.preprocessor
        )
        docs.append(doc)

    return docs


get_companies_docs(None, "train")