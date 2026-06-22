import json
import pandas as pd



with open(
    "outputs/llm_risk_results.json"
) as f:

    data = json.load(f)



df = pd.DataFrame(data)


# fix risk category lists
df["risk_category"] = df["risk_category"].apply(
    lambda x: x[0] if isinstance(x, list) else x
)


# fix severity lists
df["severity_score"] = df["severity_score"].apply(
    lambda x: x[0] if isinstance(x, list) else x
)


# convert severity to number
df["severity_score"] = pd.to_numeric(
    df["severity_score"],
    errors="coerce"
)


# remove bad rows
df = df.dropna(
    subset=[
        "risk_category",
        "severity_score"
    ]
)


summary = (
    df
    .groupby("risk_category")
    .agg(
        frequency=("risk_category","count"),
        avg_severity=("severity_score","mean")
    )
)



summary["risk_score"] = (
    summary["frequency"]
    *
    summary["avg_severity"]
)



summary = summary.sort_values(
    "risk_score",
    ascending=False
)



print(summary)



summary.to_csv(
    "outputs/risk_scores.csv"
)