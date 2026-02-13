import re
import pandas as pd


def clean_text(text: str) -> str:
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"\b\d{7,}\b", "", text)

    text = re.sub(r"[^a-z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["message"] = df["message"].apply(clean_text)

    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    return df
