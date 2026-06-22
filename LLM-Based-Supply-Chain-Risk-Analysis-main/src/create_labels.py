import pandas as pd
import json
from dotenv import load_dotenv
from google import genai
import os
import time

load_dotenv()

os.makedirs("outputs", exist_ok=True)
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


df = pd.read_csv("data/external_sentiment_news.csv")


def label_article(text):

    prompt = f"""
You are a supply chain risk labeling system.

Return ONLY valid JSON (no markdown, no explanation).

Schema:
{{
  "risk_category": "",
  "supply_chain_stage": "",
  "severity_score": 1,
  "sentiment": "",
  "root_cause": "",
  "explanation": ""
}}

Text:
{text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        raw = response.text.strip()

        # remove ```json if present
        if "```" in raw:
            raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)

    except Exception as e:
        print("JSON parse failed:", e)
        print("Raw output:", response.text)
        return None


labeled_data = []

for i, row in df.iterrows():

    if i % 10 == 0:
        print("Checkpoint:", i)

    try:
        text = row["Article Text"]
        if pd.isna(text):
            continue

        label = label_article(str(text))

        if label:
            label["text"] = text
            labeled_data.append(label)

    except Exception as e:
        print("Error at row", i, e)
        continue


with open("outputs/labeled_data.json", "w") as f:
    json.dump(labeled_data, f, indent=2)

print("Done!")