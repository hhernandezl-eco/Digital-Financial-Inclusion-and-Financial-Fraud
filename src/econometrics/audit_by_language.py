"""Deployed precision by language with Wilson 95% intervals.
Input: CSV with one row per audited post (n=400) and columns  language ('pt'/'es')  and  confirmed (1/0)."""
import sys, pandas as pd
from statsmodels.stats.proportion import proportion_confint
d = pd.read_csv(sys.argv[1])
for lang, g in list(d.groupby("language")) + [("all", d)]:
    k, n = int(g.confirmed.sum()), len(g); lo, hi = proportion_confint(k, n, method="wilson")
    print(f"{lang}: {k}/{n} = {k/n:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")
