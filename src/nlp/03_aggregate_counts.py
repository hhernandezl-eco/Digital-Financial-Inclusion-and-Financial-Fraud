"""
Step 3. From posts to publishable counts.

Input : data/private/corpus.parquet and data/private/classified.csv (one row per post, same order, with the columns
        country and typology; country comes from the profile-location step and typology from the cluster mapping).
Output: data/public/frisk/monthly_counts_by_typology.csv  -- posts per country, month, and typology.

This is the only step that moves from private to public data. The output holds counts only: no text, identifier, or
user information, and no date finer than the month.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
corpus = pd.read_parquet(ROOT / "data" / "private" / "corpus.parquet")
lab = pd.read_csv(ROOT / "data" / "private" / "classified.csv")
df = pd.concat([corpus[["created_at"]].reset_index(drop=True), lab[["country", "typology"]]], axis=1).dropna()
df["year"], df["month"] = df.created_at.dt.year, df.created_at.dt.month
out = df.groupby(["country", "year", "month", "typology"]).size().rename("posts").reset_index()
out.to_csv(ROOT / "data" / "public" / "frisk" / "monthly_counts_by_typology.csv", index=False)
print(out.groupby("country").posts.sum())
