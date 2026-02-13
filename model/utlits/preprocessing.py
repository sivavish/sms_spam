import pandas as pd
import numpy as np

df = pd.read_csv('data\spam.csv')

df = df[["v1", "v2"]]
df.columns = ["label", "message"]

print("\nMissing Values:")
print(df.isnull().sum())

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")