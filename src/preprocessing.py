
import re
import pandas as pd

def clean_text(text):
    """
    Basic text cleaning.
    """

    text = str(text)

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prepare_dataframe(df, label_map):
    """
    Prepare a dataframe for model input.

    Parameters
    ----------
    df : pandas.DataFrame

    label_map : dict

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()

    df["Topic"] = df["Topic"].fillna("")

    df["News"] = df["News"].fillna("")

    df["text"] = (
        df["Topic"].astype(str)
        + " "
        + df["News"].astype(str)
    )

    df["text"] = df["text"].apply(clean_text)

    if "Label" in df.columns:
        df["target"] = df["Label"].map(label_map)

    return df
