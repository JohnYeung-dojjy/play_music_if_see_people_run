import pandas as pd

def load_KTH()->pd.DataFrame:
  return pd.read_csv("KTH_dataset.csv", engine="pyarrow")
  