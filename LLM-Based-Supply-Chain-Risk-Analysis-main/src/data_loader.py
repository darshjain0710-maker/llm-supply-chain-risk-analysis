import pandas as pd


def load_reviews(path):

    df = pd.read_csv(path)

    df = df.dropna()

    return df["review_text"].tolist()