"""Figures 4 and 6 restricted to the estimation window (Jan-2014 onward)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frisk_lib import ROOT, OUT
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"Brazil": "#e87ba4", "Chile": "#2a78d6", "Mexico": "#eb6834", "Colombia": "#1baf7a"}
m = pd.read_csv(ROOT / "data" / "public" / "frisk" / "FRISK_canonical_monthly.csv")
m["date"] = pd.to_datetime(dict(year=m.year, month=m.month, day=1)); m = m[m.date >= "2014-01-01"]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
for c in ["Brazil", "Mexico", "Colombia", "Chile"]:
    g = m[m.country == c].set_index("date").frisk_z
    ax.plot(g.index, g, color=COL[c], lw=0.5, alpha=0.5)
    ax.plot(g.index, g.rolling(6).mean(), color=COL[c], lw=1.6, label=c)
ax.axhline(0, color="#898781", lw=0.8); ax.set_ylabel(r"FRISK index $\tilde r_{c,t}$"); ax.legend(frameon=False, ncol=4, loc="upper left")
ax.grid(axis="y", color="#e1e0d9", lw=0.5); fig.tight_layout(); fig.savefig(OUT / "Fig_Aggregated_FRISK_Consolidated.pdf")
d = pd.read_csv(ROOT / "data" / "public" / "sfc" / "colombia_official_vs_frisk.csv")
d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=1)); d = d[d.date >= "2014-01-01"].set_index("date")
y = d.ln_total_fraud_official; y = (y - y.mean()) / y.std()
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.plot(d.index, d.std_ln_FRISK_total, color="#1baf7a", lw=1.2, label="FRISK ($\mathbb{X}$ posts, Colombia)")
ax.plot(d.index, y, color="#2a78d6", lw=1.2, label="Official complaints (Superintendencia Financiera)")
ax.axhline(0, color="#898781", lw=0.8); ax.set_ylabel("Standardized index (Z-score)"); ax.legend(frameon=False)
ax.grid(axis="y", color="#e1e0d9", lw=0.5); fig.tight_layout(); fig.savefig(OUT / "Fig_Colombia_Official_Validation.pdf")
print("ok")
