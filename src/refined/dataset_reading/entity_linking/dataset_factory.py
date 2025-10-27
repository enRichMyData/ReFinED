import json
from typing import Iterable
import os
import pandas as pd

from refined.data_types.doc_types import Doc
from refined.data_types.base_types import Entity, Span
from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.resource_management.resource_manager import ResourceManager
from refined.doc_preprocessing.wikidata_mapper import WikidataMapper


class Datasets:
    def __init__(self,
                 preprocessor: Preprocessor,
                 resource_manager: ResourceManager,
                 wikidata_mapper: WikidataMapper
                 ):
        self.preprocessor = preprocessor
        self.datasets_to_files = resource_manager.get_dataset_files()
        self.wikidata_mapper = wikidata_mapper

    def get_aida_docs(
            self,
            split: str,
            include_spans: bool = True,
            include_gold_label: bool = True,
            filter_not_in_kb: bool = True,
            include_mentions_for_nil: bool = True,
    ) -> Iterable[Doc]:
        split_to_name = {
            "train": "aida_train",
            "dev": "aida_dev",
            "test": "aida_test",
        }
        assert split in split_to_name, "split must be in {train, dev, test}"
        filename = self.datasets_to_files[split_to_name[split]]
        with open(filename, "r") as f:
            for line_idx, line in enumerate(f):
                line = json.loads(line)
                text = line["text"]
                spans = None
                md_spans = None
                if include_spans:
                    spans = []
                    md_spans = []
                    for span in line["spans"]:
                        if include_mentions_for_nil:
                            md_spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"]: span["start"] + span["length"]],
                                    coarse_type="MENTION"
                                )
                            )

                        titles = [
                            uri.replace("http://en.wikipedia.org/wiki/", "")
                            for uri in span["uris"]
                            if "http://en.wikipedia.org/wiki/" in uri
                        ]

                        if len(titles) == 0:
                            continue

                        title = titles[0]
                        qcode = self.wikidata_mapper.map_title_to_wikidata_qcode(title)

                        if filter_not_in_kb and (
                                qcode is None or self.wikidata_mapper.wikidata_qcode_is_disambiguation_page(qcode)
                        ):
                            continue

                        if not filter_not_in_kb and qcode is None:
                            qcode = "Q0"

                        if not include_mentions_for_nil:
                            md_spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"]: span["start"] + span["length"]],
                                    coarse_type="MENTION"
                                )
                            )

                        if include_gold_label:
                            spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"]: span["start"] + span["length"]],
                                    gold_entity=Entity(wikidata_entity_id=qcode, wikipedia_entity_title=title),
                                    coarse_type="MENTION"
                                )
                            )
                        else:
                            spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"]: span["start"] + span["length"]],
                                    coarse_type="MENTION"
                                )
                            )

                if spans is None:
                    yield Doc.from_text(
                        text=text,
                        preprocessor=self.preprocessor
                    )
                else:
                    yield Doc.from_text_with_spans(
                        text=text, spans=spans, md_spans=md_spans, preprocessor=self.preprocessor
                    )

    def _read_standard_format(
            self,
            filename: str,
            include_spans: bool = True,
            include_gold_label: bool = True,
            filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        with open(filename, "r") as f:
            for line_idx, line in enumerate(f):
                line = json.loads(line)
                text = line["text"]
                spans = None
                md_spans = None
                if include_spans:
                    spans = []
                    md_spans = []
                    for span in line["mentions"]:
                        title = span["wiki_name"]
                        md_spans.append(
                            Span(
                                start=span["start"],
                                ln=span["length"],
                                text=text[span["start"]: span["start"] + span["length"]],
                                coarse_type="MENTION"
                            )
                        )

                        if title is None or title == "NIL":
                            continue

                        title = title.replace(" ", "_")
                        qcode = self.wikidata_mapper.map_title_to_wikidata_qcode(title)

                        if filter_not_in_kb and (
                                qcode is None or self.wikidata_mapper.wikidata_qcode_is_disambiguation_page(qcode)
                        ):
                            continue

                        if not filter_not_in_kb and qcode is None:
                            qcode = "Q0"

                        if include_gold_label:
                            spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"]: span["start"] + span["length"]],
                                    gold_entity=Entity(wikidata_entity_id=qcode, wikipedia_entity_title=title),
                                    coarse_type="MENTION"
                                )
                            )
                        else:
                            spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"]: span["start"] + span["length"]],
                                    coarse_type="MENTION"
                                )
                            )
                if spans is None:
                    yield Doc.from_text(
                        text=text,
                        preprocessor=self.preprocessor
                    )
                else:
                    yield Doc.from_text_with_spans(
                        text=text, spans=spans, md_spans=md_spans, preprocessor=self.preprocessor
                    )

    def get_msnbc_docs(
            self,
            split: str,
            include_spans: bool = True,
            include_gold_label: bool = True,
            filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split == "test", "MSNBC only has a test dataset"
        return self._read_standard_format(
            filename=self.datasets_to_files['msnbc'],
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_aquaint_docs(
            self,
            split: str,
            include_spans: bool = True,
            include_gold_label: bool = True,
            filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split == "test", "aquaint only has a test dataset"
        return self._read_standard_format(
            filename=self.datasets_to_files['aquaint'],
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_ace2004_docs(
            self,
            split: str,
            include_spans: bool = True,
            include_gold_label: bool = True,
            filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split == "test", "ace2004 only has a test dataset"
        return self._read_standard_format(
            filename=self.datasets_to_files['ace2004'],
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_cweb_docs(
            self,
            split: str,
            include_spans: bool = True,
            include_gold_label: bool = False,
            filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split == "test", "cweb only has a test dataset"
        return self._read_standard_format(
            filename=self.datasets_to_files['clueweb'],
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_wiki_docs(
            self,
            split: str,
            include_spans: bool = True,
            include_gold_label: bool = False,
            filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split == "test", "wiki only has a test dataset"
        return self._read_standard_format(
            filename=self.datasets_to_files['wikipedia'],
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_webqsp_docs(
            self,
            split: str,
            include_spans: bool = True,
            include_gold_label: bool = True,
            filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split in {"train", "dev", "test"}, "webqsp has train/dev/test splits."
        file_name = self.datasets_to_files[{"train": "webqsp_train_data_el",
                                            "dev": "webqsp_dev_data_el",
                                            "test": "webqsp_test_data_el"
                                            }[split]]
        with open(file_name, 'r') as f:
            for dataset_line in f:
                dataset_line = json.loads(dataset_line)
                text = dataset_line["text"]
                dataset_spans = [
                    {
                        "text": text[mention[0]: mention[1]],
                        "start": mention[0],
                        "end": mention[1],
                        "qcode": qcode,
                    }
                    for mention, qcode in zip(dataset_line["mentions"], dataset_line["wikidata_id"])
                ]
                dataset_spans.sort(key=lambda x: x["start"])
                spans = []
                md_spans = []
                for dataset_span in dataset_spans:
                    md_spans.append(
                        Span(
                            start=dataset_span["start"],
                            ln=dataset_span["end"] - dataset_span["start"],
                            text=dataset_span["text"],
                            coarse_type="MENTION"  # All entity types are "MENTION"s in WebQSP (no numerics).
                        )
                    )
                    spans.append(
                        Span(
                            start=dataset_span["start"],
                            ln=dataset_span["end"] - dataset_span["start"],
                            text=dataset_span["text"],
                            gold_entity=Entity(
                                wikidata_entity_id=dataset_span["qcode"]) if include_gold_label else None,
                            coarse_type="MENTION"
                        )
                    )
                yield Doc.from_text_with_spans(text=text, spans=spans, preprocessor=self.preprocessor,
                                               md_spans=md_spans)

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
        # -------------------- Load data --------------------
        # Paths
        text_path = f"my_tests/data/movies_{split}.csv"
        truth_path = f"my_tests/data/el_movies_gt_wikidata.csv"

        # Load CSVs
        dftext = pd.read_csv(text_path)
        dftruth = pd.read_csv(truth_path)
        dftruth = dftruth[dftruth["tableName"] == f"movies_{split}"]  # filter to relevant split
        dftruth["entity"] = dftruth["entity"].str.replace("http://www.wikidata.org/entity/", "", regex=False)  # only use QID

        # Merge text and truth by row index
        frame = pd.merge(dftext, dftruth, left_index=True, right_on="idRow")

        docs = []
        for _, row in frame.iterrows():
            # Full text line as context
            text = ",".join(str(row[col]) for col in dftext.columns if pd.notna(row[col]))

            # Gold entity if available
            gold_entity = None
            if include_gold_label and pd.notna(row["entity"]):
                gold_entity = Entity(
                    wikidata_entity_id=row["entity"],
                    wikipedia_entity_title=row["Title"].replace(" ", "_"),
                    human_readable_name=row["Title"]
                )

            start = text.find(str(row['Title']))
            length = len(str(row['Title']))

            spans, md_spans = [], []
            if include_spans and start != -1:
                # Gold span for main entity
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

            # Build doc
            doc = Doc.from_text_with_spans(
                text=text,
                spans=spans,
                md_spans=md_spans,
                preprocessor=self.preprocessor
            )
            docs.append(doc)

        return docs

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
        # -------------------- Load data --------------------
        # Paths
        text_path =  f"my_tests/data/companies/companies_{split}.csv"       # train/test split
        truth_path = f"my_tests/data/companies/el_companies_gt_wikidata.csv"

        # Load CSVs
        dftext = pd.read_csv(text_path)
        dftruth = pd.read_csv(truth_path)
        dftruth = dftruth[dftruth["tableName"] == f"companies_{split}"]  # train/test split
        dftruth["entity"] = dftruth["entity"].str.replace("http://www.wikidata.org/entity/", "",
                                                          regex=False)  # only use QID
        # Merge text and truth by row index in one frame
        frame = pd.merge(dftext, dftruth, left_index=True, right_on="idRow")

        docs = []
        for _, row in frame.iterrows():
            # Full text line as context
            text = ",".join(str(row[col]) for col in dftext.columns if pd.notna(row[col]))

            # Gold entity if available
            gold_entity = None
            if include_gold_label and pd.notna(row["entity"]):
                gold_entity = Entity(
                    wikidata_entity_id=row["entity"],
                    wikipedia_entity_title=row["company"].replace(" ", "_"),
                    human_readable_name=row["company"]
                )

            # start and length of main entity
            start = text.find(str(row['company']))
            length = len(str(row['company']))

            spans, md_spans = [], []
            if include_spans and start != -1:
                # Gold span for main entity
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

            # Build doc
            doc = Doc.from_text_with_spans(
                text=text,
                spans=spans,
                md_spans=md_spans,
                preprocessor=self.preprocessor
            )
            docs.append(doc)

        return docs
