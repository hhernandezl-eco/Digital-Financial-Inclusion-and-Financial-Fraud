"""make_figures.py -- figures of the revised application (V2). Reads output/results.pkl."""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frisk_est import *

R = pickle.load(open(OUT / "results.pkl", "rb"))
panel = load_panel()

# fixed colour per entity (first three slots of the validated categorical palette), plus marker as secondary encoding
COL = {"Chile": "#2a78d6", "Mexico": "#eb6834", "Colombia": "#1baf7a"}
MRK = {"Chile": "o", "Mexico": "s", "Colombia": "^"}
INK, MUTED, GRID_C = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#898781", "axes.linewidth": 0.6, "axes.titlesize": 9,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def style(ax, zero=True):
    ax.grid(axis="y", color=GRID_C, lw=0.5)
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0, color="#898781", lw=0.8)


# ---------------------------------------------------------------------------------------------
# Figure A: impulse responses (local projections), by country
# ---------------------------------------------------------------------------------------------
lp = R["lp"]
beta, se_all = lp["beta"], np.sqrt(np.diag(lp["V"])).reshape(3, H + 1)
fsb, fss = R["first_stage"]
h = np.arange(H + 1)
fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4), sharex=True)
for j, c in enumerate(COUNTRIES):
    col = COL[c]
    # row 1: response of FRISK itself
    ax = axes[0, j]
    style(ax)
    lo, hi = fsb[j] - 1.96 * fss[j], fsb[j] + 1.96 * fss[j]
    ax.fill_between(h[1:], lo[1:], hi[1:], color=col, alpha=0.20, lw=0)
    ax.plot(h, fsb[j], color=col, lw=1.4, marker=MRK[c], ms=3.5, mfc="white", mew=1.0)
    ax.set_title(c, color=INK, fontweight="bold")
    ax.set_ylim(-0.6, 1.25)
    if j == 0:
        ax.set_ylabel("FRISK index\n(units of $\\tilde r$)")
    # row 2: response of IFI
    ax = axes[1, j]
    style(ax)
    sc = R["sup"][c]["crit"]
    b, s = beta[j] * 1e3, se_all[j] * 1e3
    ax.fill_between(h, b - sc * s, b + sc * s, color=col, alpha=0.12, lw=0)
    ax.fill_between(h, b - 1.96 * s, b + 1.96 * s, color=col, alpha=0.25, lw=0)
    ax.plot(h, b, color=col, lw=1.4, marker=MRK[c], ms=3.5, mfc="white", mew=1.0)
    ax.set_xlabel("Months after the shock ($h$)")
    ax.set_xticks(h)
    sm = R["lp_sum"][(c, "avg")]
    ax.text(0.04, 0.05, f"avg. $h$=1--6: {sm['est_sd']:+.2f} s.d. ({sm['se_sd']:.2f})".replace("-0", "−0"), transform=ax.transAxes,
            fontsize=7, color=MUTED, ha="left", va="bottom")
    if j == 0:
        ax.set_ylabel("IFI response\n(10$^{-3}$ IFI points)")
for ax in axes[1]:
    ymin = min(ax.get_ylim()[0] for ax in axes[1]); ymax = max(ax.get_ylim()[1] for ax in axes[1])
for j in range(3):
    pass
fig.tight_layout(h_pad=1.0, w_pad=0.8)
fig.savefig(OUT / "Fig_LP_IRF.pdf")
fig.savefig(OUT / "Fig_LP_IRF.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------------------------
# Figure B (appendix): IFI levels, FRISK levels, and monthly changes in IFI
# ---------------------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(6.6, 5.6), sharex="col")
for j, c in enumerate(COUNTRIES):
    g = panel[c]
    col = COL[c]
    ax = axes[0, j]; style(ax, zero=False)
    ax.plot(g.index, g.ifi, color=col, lw=1.2)
    o = g.ifi_pooled.dropna(); ni = g.ifi.dropna()
    ax.plot(o.index, (o - o.min()) / (o.max() - o.min()) * (ni.max() - ni.min()) + ni.min(), color="#898781", lw=0.9, ls="--", alpha=0.8)
    ax.set_title(c, fontweight="bold")
    if c == "Colombia":
        ax.axvspan(pd.Timestamp(COL_BREAK[0]), pd.Timestamp(COL_BREAK[1]), color="#898781", alpha=0.18, lw=0)
        ax.text(pd.Timestamp("2022-05-01"), ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "Jul-2022 to\nJan-2024", fontsize=6.5,
                color=MUTED, va="bottom", ha="right")
    if j == 0:
        ax.set_ylabel("IFI (within-country)")
    ax = axes[1, j]; style(ax)
    ax.plot(g.index, g.r, color=col, lw=0.6, alpha=0.55)
    ax.plot(g.index, g.r.rolling(6).mean(), color=col, lw=1.4)
    if j == 0:
        ax.set_ylabel("FRISK $\\tilde r$\n(thin: monthly; thick: 6-m. MA)")
    ax = axes[2, j]; style(ax)
    ax.plot(g.index, g.ifi.diff() * 1e3, color=col, lw=0.8)
    if c == "Colombia":
        ax.axvspan(pd.Timestamp(COL_BREAK[0]), pd.Timestamp(COL_BREAK[1]), color="#898781", alpha=0.18, lw=0)
    if j == 0:
        ax.set_ylabel("$\\Delta$IFI (10$^{-3}$)")
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(5))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
fig.tight_layout(h_pad=0.8, w_pad=0.8)
fig.savefig(OUT / "Fig_Series_Levels_Diffs.pdf")
fig.savefig(OUT / "Fig_Series_Levels_Diffs.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------------------------
# Figure C (appendix): robustness of the average response across specifications
# ---------------------------------------------------------------------------------------------
rob = R["robust"]
labels = [s.replace("$", "").replace("\\Delta IFI", "dIFI").replace("\\", "") for s in rob.spec]
fig, axes = plt.subplots(1, 4, figsize=(6.8, 3.6), sharey=True)
names = COUNTRIES + ["MG"]
titles = COUNTRIES + ["Mean group"]
y = np.arange(len(rob))[::-1]
for j, nm in enumerate(names):
    ax = axes[j]
    col = COL.get(nm, "#52514e")
    mk = MRK.get(nm, "D")
    est, se = rob[f"{nm}_est"].values, rob[f"{nm}_se"].values
    ax.axvline(0, color="#898781", lw=0.8)
    ax.grid(axis="x", color=GRID_C, lw=0.5)
    ax.hlines(y, est - 1.96 * se, est + 1.96 * se, color=col, lw=1.2)
    ax.plot(est, y, mk, color=col, ms=3.8, mfc="white", mew=1.0)
    ax.set_title(titles[j], fontweight="bold")
    ax.set_xlabel("s.d. of monthly $\\Delta$IFI")
    ax.set_ylim(-0.7, len(rob) - 0.3)
axes[0].set_yticks(y)
short = ["Baseline", "$p=1$", "$p=6$", "No month dummies", "Macro controls", "Original pooled IFI",
         "FRISK 3-m. MA", "Outliers trimmed", "Excl. Mar-Dec 2020", "Colombia: excl. break", "Lag-augmented, White",
         "FRISK in levels + trend", "Controls + inflation"]
axes[0].set_yticklabels(short, fontsize=7)
fig.tight_layout(w_pad=0.6)
fig.savefig(OUT / "Fig_Robustness_Avg.pdf")
fig.savefig(OUT / "Fig_Robustness_Avg.png", dpi=200)
plt.close(fig)
print("figures written")
