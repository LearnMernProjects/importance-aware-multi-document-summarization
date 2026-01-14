import pandas as pd

csv_path = "data/processed/newssumm_raw.csv"

df = pd.read_csv(csv_path)

print("\n CSV validation summary")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Column names:", df.columns.tolist())

print("\nSample rows:")
print(df.head(3))
