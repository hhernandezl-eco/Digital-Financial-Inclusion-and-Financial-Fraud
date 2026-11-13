"""
Builds the FRISK index from monthly post counts and population.

    FRISK_raw = 100,000 * posts / population      (population in thousands -> posts * 100 / pop_thousands)
    r         = log(1 + FRISK_raw)
    FRISK_z   = (r - mean) / sd                    (mean and sd over the pooled Chile-Mexico-Colombia panel)

Input : data/public/frisk/monthly_counts.csv   (country, year, month, total_pop_thousands, posts)
Output: data/derived/frisk_index.csv

The counts are the only information about the corpus that the estimation needs, so the whole econometric part of the
project can be reproduced without the posts themselves.
"""
import numpy as np
import pandas as pd
from frisk_lib import ROOT, DATA

PANEL = ["Chile", "Mexico", "Colombia"]
m = pd.read_csv(ROOT / "data" / "public" / "frisk" / "monthly_counts.csv")
m["frisk_raw"] = 100 * m.posts / m.total_pop_thousands
m["r"] = np.log1p(m.frisk_raw)
ref = m[m.country.isin(PANEL)].r
m["frisk_z"] = (m.r - ref.mean()) / ref.std(ddof=0)          # Brazil is scored with the panel mean and sd
DATA.mkdir(parents=True, exist_ok=True)
m.to_csv(DATA / "frisk_index.csv", index=False)

check = pd.read_csv(ROOT / "data" / "public" / "frisk" / "FRISK_canonical_monthly.csv")
diff = (m.frisk_z.values - check.frisk_z.values)
print("max |difference| with the stored index:", float(np.abs(diff).max()))
