"""
Turns the keyword dictionary (data/public/keywords/keywords.xlsx, sheet 'Diccionario_Total') into search queries.

The search endpoint limits the length of a query, so the terms are grouped in blocks that are joined with OR. Every block is
restricted to the requested language and excludes retweets. Multi-word terms are quoted.
"""
import pandas as pd

MAX_LEN = 900          # characters per query, below the endpoint limit


def build_queries(path: str, lang: str, sheet: str = "Diccionario_Total"):
    terms = pd.read_excel(path, sheet_name=sheet).iloc[:, 0].dropna().astype(str).str.strip().unique()
    terms = [f'"{t}"' if " " in t else t for t in terms if t]
    suffix = f" lang:{lang} -is:retweet"
    blocks, cur = [], []
    for t in terms:
        trial = "(" + " OR ".join(cur + [t]) + ")" + suffix
        if len(trial) > MAX_LEN and cur:
            blocks.append("(" + " OR ".join(cur) + ")" + suffix)
            cur = [t]
        else:
            cur.append(t)
    if cur:
        blocks.append("(" + " OR ".join(cur) + ")" + suffix)
    return blocks


if __name__ == "__main__":
    import sys
    qs = build_queries(sys.argv[1], sys.argv[2])
    print(len(qs), "queries")
    print(qs[0])
