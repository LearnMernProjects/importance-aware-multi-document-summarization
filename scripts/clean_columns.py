import pandas as pd

input_csv = "data/processed/newssumm_raw.csv"
output_csv = "data/processed/newssumm_clean.csv"

df = pd.read_csv(input_csv)

# Keep only meaningful columns
valid_columns = [
    "newspaper_name",
    "published_date\n",
    "headline",
    "article_text",
    "human_summary",
    "news_category"
]

df = df[valid_columns]

# Clean column name
df = df.rename(columns={"published_date\n": "published_date"})

df.to_csv(output_csv, index=False)

print("Cleaned CSV saved.")
print("Columns:", df.columns.tolist())
print("Rows:", len(df))
