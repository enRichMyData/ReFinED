import pandas as pd


def process_csv(file_path):
    df = pd.read_csv(file_path)

    print(f'Loaded {len(df)} rows with columns: {list(df.columns)}')

    # NB! bruker bare "tittel" "relase year" og "overview" for øyeblikket
    # er totalt 11 kolonner
    processed_texts = [
        f"{row['Series_Title']} released in {row['Released_Year']}. {row['Overview']}"
        for _, row in df.iterrows()
    ]

    return processed_texts


if __name__ == "__main__":
    texts = process_csv("data/imdb_top_100.csv")
    print("\n--- First 5 Combined Texts ---")
    for t in texts[:5]:
        print(t)