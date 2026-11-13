"""Merge the two Monte Carlo designs (run in parallel) into output/mc_size2.csv."""
import pandas as pd
from frisk_lib import OUT
pd.concat([pd.read_csv(OUT / "mc_size2_A.csv"), pd.read_csv(OUT / "mc_size2_B.csv")]).to_csv(OUT / "mc_size2.csv", index=False)
print(pd.read_csv(OUT / "mc_size2.csv").T)
