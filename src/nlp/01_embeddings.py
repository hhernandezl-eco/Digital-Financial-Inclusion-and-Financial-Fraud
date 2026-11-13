"""
Step 1. Sentence embeddings with mDeBERTa-v3-base.

Reads the collected posts (data/private/raw/{es,pt}/*.jsonl, produced by src/collection/collect_posts.py), lowercases the text
and stores, for every batch of posts, the hidden state of the first token ([CLS]) of the last layer of the pretrained
microsoft/mdeberta-v3-base encoder (768 dimensions, sequences truncated to 128 tokens).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

PRIVATE = Path(__file__).resolve().parents[2] / "data" / "private"
MODEL = "microsoft/mdeberta-v3-base"


def load_corpus(dir_pt: Path, dir_es: Path) -> pd.DataFrame:
    rows = []
    for lang, d in (("pt", dir_pt), ("es", dir_es)):
        for f in sorted(d.glob("*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                rows += [dict(json.loads(line), lang=lang) for line in fh]
    df = pd.DataFrame(rows).dropna(subset=["text"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["texto_limpio"] = df["text"].astype(str).str.lower().str.strip()
    return df.drop_duplicates("post_id").reset_index(drop=True)


def embed(textos: pd.Series, out_dir: Path, batch: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = AutoTokenizer.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).to(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    lista = textos.tolist()
    for i in tqdm(range(0, len(lista), batch), desc="embeddings"):
        enc = tok(lista[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
        np.save(out_dir / f"emb_{i:08d}.npy", h)


if __name__ == "__main__":
    df = load_corpus(PRIVATE / "raw" / "pt", PRIVATE / "raw" / "es")
    df[["post_id", "created_at", "lang", "profile_location", "texto_limpio"]].to_parquet(PRIVATE / "corpus.parquet")
    embed(df["texto_limpio"], PRIVATE / "embeddings")
