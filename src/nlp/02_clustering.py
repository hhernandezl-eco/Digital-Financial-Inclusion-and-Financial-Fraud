"""
Step 2. Density-based clustering of the embeddings of the posts flagged as fraud complaints.

DBSCAN runs directly on the 768-dimensional embeddings (no dimensionality reduction), with cosine distance,
eps = 0.5 and min_samples = 5 (Section 3.4 of the paper). Posts that DBSCAN leaves as noise get the label -1.

Input : data/private/embeddings/*.npy (step 1) and data/private/predictions.csv (step 1b; column `fraud`, one row per post,
        same order as corpus.parquet).
Output: data/private/clusters.csv with columns  row (position in corpus.parquet) and  cluster.

Clusters are then mapped by hand to the typologies of the paper after inspecting their most frequent terms.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

PRIVATE = Path(__file__).resolve().parents[2] / "data" / "private"
EPS, MIN_SAMPLES = 0.5, 5


def load_embeddings(dir_emb: Path) -> np.ndarray:
    return np.vstack([np.load(f) for f in sorted(dir_emb.glob("*.npy"))]).astype(np.float32)


def cluster(X: np.ndarray) -> np.ndarray:
    return DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="cosine", algorithm="brute", n_jobs=-1).fit_predict(X)


if __name__ == "__main__":
    X = load_embeddings(PRIVATE / "embeddings")
    fraud = pd.read_csv(PRIVATE / "predictions.csv")["fraud"].to_numpy().astype(bool)
    assert len(fraud) == len(X), "predictions.csv and the embeddings must have one row per post"
    rows = np.flatnonzero(fraud)
    pd.DataFrame({"row": rows, "cluster": cluster(X[rows])}).to_csv(PRIVATE / "clusters.csv", index=False)
