import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

import joblib



# load Gemini labelled data

with open(
    "outputs/llm_risk_results.json",
    "r"
) as f:

    data=json.load(f)



df=pd.DataFrame(data)


print(df.head())



X=df["text"]

y=df["risk_category"]



model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )

])



model.fit(
    X,
    y
)


joblib.dump(
    model,
    "outputs/risk_classifier.pkl"
)


print("Model trained!")