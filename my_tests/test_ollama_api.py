import os
from openai import OpenAI

# Example movie rows as CSV-style strings
movie_lines = [
    "La Femme du cosmonaute,1998-01-01,,comedy film,France,,Jacques Monnet",
    "Gods of the Plague,1969-01-01,91.0,drama film,Germany,German,Rainer Werner Fassbinder",
    # "The Company Men,2010-01-01,109.0,drama film,United States of America,English,John Wells",
    # "Visible Secret,2001-01-01,98.0,horror film,Hong Kong,Cantonese,Ann Hui",
]

# Column index to get QID for
col_index = 0  # Title column

def get_qid_from_llm(line, col_index, client):
    cells = line.split(",")
    cell_value = cells[col_index]
    prompt = (
        f"You are a Wikidata expert. Given the CSV row:\n"
        f"{line}\n"
        f"Return the Wikidata QID of the value in column {col_index} ('{cell_value}') "
        f"ONLY. No extra text, explanation, or quotes."
    )
    response = client.chat.completions.create(
        model="gpt-oss:120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# dette er ikke veldig bra

#TODO:
# for bruke token i pycharm:
# > edit configurations
# -> environment variables:
# ---> "OLLAMA_API_KEY=your_token_here"


def main():
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is not set in the environment")

    client = OpenAI(
        base_url="https://ollama.sct.sintef.no/v1",
        api_key=api_key,
    )

    for i, line in enumerate(movie_lines):
        qid = get_qid_from_llm(line, col_index, client)
        print(f"[{i+1}] {line} -> Column {col_index} QID: {qid}")

if __name__ == "__main__":
    main()



