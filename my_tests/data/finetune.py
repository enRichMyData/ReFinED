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

def get_wikidata_docs(frame) -> Iterable[Doc]:
    docs = []
    for _, row in frame.iterrows():
        text = f"{row['Title']},{row['Release_Date']},{row['Duration']},{row['Genre']},{row['Country']},{row['Language']},{row['Director']}"

        gold_entity = row['entity']

        print(f"{'Text:':<15}{text}")
        print(f"{'Gold Entity:':<15}{gold_entity}")
        break

get_wikidata_docs(frame)