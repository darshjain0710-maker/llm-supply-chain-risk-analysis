# src/preprocess.py
import pandas as pd

def clean_text(text):
    if pd.isna(text):
        return None

    text = str(text)
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.strip()

    return text