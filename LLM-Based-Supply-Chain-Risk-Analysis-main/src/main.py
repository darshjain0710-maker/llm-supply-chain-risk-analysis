import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
from dotenv import load_dotenv

from llm_extractor import extract_risk


load_dotenv()



df = pd.read_csv(
    "data/labeled_logistics_news.csv"
)


results=[]


def process_article(row):

    output = extract_risk(
        row["clean_text"]
    )

    if isinstance(output, list):
        output = output[0]

    output["text"] = row["clean_text"]

    return output



rows = [
    row for _, row in df.iterrows()
]


results = []


with ThreadPoolExecutor(max_workers=5) as executor:

    for result in tqdm(
        executor.map(process_article, rows),
        total=len(rows)
    ):

        results.append(result)



with open(
    "outputs/llm_risk_results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("LLM extraction completed")



with open(
    "outputs/llm_risk_results.json",
    "w"
) as f:


    json.dump(
        results,
        f,
        indent=4
    )



print(
    "LLM extraction completed"
)