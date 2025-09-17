import pandas as pd

def process_csv(file_path):
    df = pd.read_csv(file_path)

    print(f'Loaded {len(df)} rows with columns: {list(df.columns)}')

    # Use for IMDb
    # if "Series_Title" in df.columns:
    #     processed_texts = [
    #         (
    #             f"{row['Series_Title']} released in {row['Released_Year']}.\n"
    #             f"Overview: {row['Overview']}\n"
    #             f"Directed by {row['Director']}.\n"
    #             f"Starring: {row['Star1']}.\n"
    #             f"Genre: {row['Genre']}.\n"
    #             f"Runtime: {row['Runtime (min)']} minutes.\n"
    #             f'"IMDb" rating: {row["IMDB_Rating"]}, "Metascore": {row["Meta_score"]}.\n'
    #             f"Votes: {row['No_of_Votes']}, Gross: {row['Gross']}."
    #         )
    #         for _, row in df.iterrows()
    #     ]
    #
    # # Used for company file
    # elif "company" in df.columns:
    #     processed_texts = [
    #         (
    #             f"Company: {row['company']}\n"
    #             f"Founded: {row['Founded Year']}\n"
    #             f"Website: {row['Website']}\n"
    #             f"Twitter: {row['X (Twitter)']}\n"
    #             f"Employees: {row['Employees']}\n"
    #             f"Coordinates: {row['Coordinates']}\n"
    #             f"Founded By: {row['Founded By']}\n"
    #             f"Industry: {row['Industry']}\n"
    #             f"Country: {row['Country']}\n"
    #             f"Headquarters: {row['Headquarters']}"
    #         )
    #         for _, row in df.iterrows()
    #     ]
    #
    # # Generic use for unknown CSVs
    # else:
    processed_texts = [
        ", ".join(str(row[col]) for col in df.columns if pd.notna(row[col]))
        for _, row in df.iterrows()
    ]

    return processed_texts

if __name__ == "__main__":
    texts = process_csv("data/imdb_top_100.csv")
    print("\n--- First 5 Combined Texts ---")
    for t in texts[:5]:
        print(t)
