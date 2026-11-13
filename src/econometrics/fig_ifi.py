"""Rebuilt-IFI figures: main-text Figure 7 (IFI by country) and update of the series figure (appendix) with the original pooled series overlaid."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frisk_lib import *
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"Chile": "#2a78d6", "Mexico": "#eb6834", "Colombia": "#1baf7a"}
P = load_panel()
fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.4), sharey=True)
for j, c in enumerate(COUNTRIES):
    ax[j].plot(P[c].index, P[c].ifi, color=COL[c], lw=1.3); ax[j].set_title(c, fontweight="bold")
    ax[j].grid(axis="y", color="#e1e0d9", lw=0.5); ax[j].xaxis.set_major_locator(matplotlib.dates.YearLocator(5)); ax[j].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax[0].set_ylabel("IFI (normalized within country)"); fig.tight_layout(); fig.savefig(OUT / "Fig_IFI_TimeSeries.pdf"); print("ok")
