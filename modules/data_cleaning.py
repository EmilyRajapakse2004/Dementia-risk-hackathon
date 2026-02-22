import pandas as pd

def load_and_clean_data(filepath):
    """
    Load CSV and remove missing values
    """
    df = pd.read_csv(filepath)
    df = df.dropna().reset_index(drop=True)
    return df