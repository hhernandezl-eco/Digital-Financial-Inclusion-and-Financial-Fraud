"""
Step 2. Dimensionality reduction and density-based clustering of the embeddings.

1. Incremental PCA to 50 components over the stored embeddings.
2. Standardization of the components.
3. DBSCAN (eps = 0.5, min_samples = 10, Euclidean distance) on a random sample of 50,000 posts.
4. The remaining posts receive the label of their nearest neighbor in the sample (1-NN).

Writes data/private/clusters.csv with one row per post (same order as corpus.parquet).
Clusters are then mapped by hand to the typologies of the paper after inspecting their most frequent terms.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PRIVATE = Path(__file__).resolve().parents[2] / "data" / "private"
SEED = 0


def reduce(dir_emb: Path, comps: int = 50) -> np.ndarray:
    files = sorted(dir_emb.glob("*.npy"))
    ipca = IncrementalPCA(n_components=comps)
    for f in files:
        ipca.partial_fit(np.load(f))
    return np.vstack([ipca.transform(np.load(f)).astype(np.float32) for f in files])


def cluster(X: np.ndarray, sample: int = 50000) -> np.ndarray:
    Z = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Z), min(sample, len(Z)), replace=False)
    lab = DBSCAN(eps=0.5, min_samples=10, metric="euclidean", n_jobs=-1).fit_predict(Z[idx])
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(Z[idx], lab)
    return knn.predict(Z)


if __name__ == "__main__":
    X = reduce(PRIVATE / "embeddings")
    pd.DataFrame({"cluster": cluster(X)}).to_csv(PRIVATE / "clusters.csv", index=False)
