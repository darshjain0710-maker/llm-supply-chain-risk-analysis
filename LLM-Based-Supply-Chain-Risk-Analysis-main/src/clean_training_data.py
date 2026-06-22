import json
import pandas as pd


with open(
    "outputs/llm_risk_results.json",
    "r"
) as f:
    data = json.load(f)


df = pd.DataFrame(data)


print("Columns available:")
print(df.columns)


# remove rows without required fields

df = df.dropna(
    subset=[
        "text",
        "risk_category"
    ]
)



# Convert list categories to single label

def clean_category(x):

    if isinstance(x, list):

        return x[0]

    return x



df["risk_category"] = df["risk_category"].apply(
    clean_category
)



# Keep only available columns

columns = [
    "text",
    "risk_category"
]


if "severity" in df.columns:
    columns.append("severity")



df = df[columns]


print("\nCleaned data:")
print(df.head())


df.to_csv(
    "outputs/training_data.csv",
    index=False
)


print(
    "\nSaved:",
    len(df),
    "samples"
)