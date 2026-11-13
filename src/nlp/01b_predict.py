"""
Step 1b. Flags first-person fraud complaints with the fine-tuned classifier (output of 00_finetune_classifier.py).

Writes data/private/predictions.csv with one row per post of corpus.parquet and the column `fraud` (0/1).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PRIVATE = Path(__file__).resolve().parents[2] / "data" / "private"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = PRIVATE / "classifier" / "final"
    tok = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    textos = pd.read_parquet(PRIVATE / "corpus.parquet")["texto_limpio"].tolist()
    pred = []
    for i in tqdm(range(0, len(textos), 64), desc="classifying"):
        enc = tok(textos[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            pred += model(**enc).logits.argmax(-1).cpu().tolist()
    pd.DataFrame({"fraud": np.array(pred)}).to_csv(PRIVATE / "predictions.csv", index=False)
