import pandas as pd
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)



# Load baseline labels

df = pd.read_csv(
    "data/labeled_logistics_news.csv"
)



# Load LLM outputs

with open(
    "outputs/llm_risk_results.json",
    "r"
) as f:

    llm_results = json.load(f)



true_labels = []

pred_labels = []



for i,item in enumerate(llm_results):


    true_labels.append(
        df.iloc[i]["risk_category"]
    )


    pred_labels.append(
        item.get(
            "risk_category",
            "No Risk"
        )
    )



print(
    "Accuracy:",
    accuracy_score(
        true_labels,
        pred_labels
    )
)


print(
    "Precision:",
    precision_score(
        true_labels,
        pred_labels,
        average="weighted",
        zero_division=0
    )
)


print(
    "Recall:",
    recall_score(
        true_labels,
        pred_labels,
        average="weighted",
        zero_division=0
    )
)


print(
    "F1 Score:",
    f1_score(
        true_labels,
        pred_labels,
        average="weighted",
        zero_division=0
    )
)



print(
    classification_report(
        true_labels,
        pred_labels,
        zero_division=0
    )
)