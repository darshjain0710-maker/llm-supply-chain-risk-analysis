import pandas as pd
import json

from preprocess import clean_text
from weak_label import weak_label

df = pd.read_csv("data/external_sentiments_news.csv")

dataset = []

for i, row in df.iterrows():

    text = clean_text(row["Article Text"])

    if not text:
        continue

    label = weak_label(text)

    dataset.append({
        "text": text,
        "risk_category": label
    })

with open("outputs/labeled_data.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Dataset created!")