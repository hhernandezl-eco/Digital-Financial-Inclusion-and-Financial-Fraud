"""Section 4.4 revisited: FRISK vs SFC complaints (Colombia, Jan-2014..Jan-2022, N=97) with time-series-valid inference."""
import numpy as np, pandas as pd, warnings
from scipy import stats
from frisk_lib import ROOT, OUT
warnings.filterwarnings("ignore")
d = pd.read_csv(ROOT / "data" / "public" / "sfc" / "colombia_official_vs_frisk.csv")
d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=1))
d = d[d.date >= "2014-01-01"].set_index("date")  # estimation window starts Jan-2014 (Dec-2013 is a pre-sample placeholder)
x = d["std_ln_FRISK_total"]; y = d["ln_total_fraud_official"]
assert len(d) == 96

def nw_p(a, b, L):
    a, b = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    X = np.column_stack([np.ones(len(a)), a.values]); yy = b.values
    bh = np.linalg.lstsq(X, yy, rcond=None)[0]; u = yy - X @ bh
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, L + 1):
        G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None]); S += (1 - l / (L + 1)) * (G + G.T)
    B = np.linalg.inv(X.T @ X); V = B @ S @ B
    t = bh[1] / np.sqrt(V[1, 1]); return t, 2 * (1 - stats.norm.cdf(abs(t)))

def neff(a, b, K=None):
    n = len(a); K = K or int(n ** (1 / 3))
    ra = [pd.Series(a).autocorr(k) for k in range(1, K + 1)]; rb = [pd.Series(b).autocorr(k) for k in range(1, K + 1)]
    return n / (1 + 2 * sum(p * q for p, q in zip(ra, rb)))

def row(name, a, b):
    m = pd.concat([a, b], axis=1).dropna(); a, b = m.iloc[:, 0], m.iloc[:, 1]
    r, p0 = stats.pearsonr(a, b); rho, ps = stats.spearmanr(a, b)
    t, ph = nw_p(a, b, 6); ne = neff(a.values, b.values)
    pe = 2 * (1 - stats.t.cdf(abs(r * np.sqrt((ne - 2) / (1 - r ** 2))), max(ne - 2, 1)))
    return dict(measure=name, N=len(m), pearson=r, p_naive=p0, t_hac=t, p_hac=ph, n_eff=ne, p_neff=pe, spearman=rho, p_spearman=ps,
                rho1_frisk=a.autocorr(1), rho1_off=b.autocorr(1))

def detr(s):
    t = np.arange(len(s)); return pd.Series(s.values - np.polyval(np.polyfit(t, s.values, 1), t), index=s.index)
rows = [row("Levels", x, y),
        row("Linearly detrended levels", detr(x), detr(y)),
        row("12-month change (annual growth)", x.diff(12), y.diff(12)),
        row("6-month moving averages, detrended", detr(x.rolling(6).mean().dropna()), detr(y.rolling(6).mean().dropna())),
        row("Monthly first differences", x.diff(), y.diff())]
T = pd.DataFrame(rows); T.to_csv(OUT / "sfc_validation.csv", index=False)
cc = []
for k in range(-6, 7):
    for nm, a, b in (("levels", x, y), ("detrended", detr(x), detr(y)), ("diff", x.diff(), y.diff())):
        m = pd.concat([a, b.shift(-k)], axis=1).dropna()
        cc.append(dict(k=k, series=nm, r=stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])[0], rho=stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1])[0], N=len(m)))
C = pd.DataFrame(cc); C.to_csv(OUT / "sfc_crosscorr.csv", index=False)
pd.set_option("display.width", 220); print(T.round(3).to_string())
print(C.pivot(index="k", columns="series", values="r").round(2).T.to_string())
print(C.pivot(index="k", columns="series", values="rho").round(2).T.to_string())

# ---- LaTeX table
def f2(v, d=3):
    return ("$-$" if v < 0 and round(abs(v), d) > 0 else "") + f"{abs(v):.{d}f}"
L = [r"\begin{tabular}{@{} L{4.4cm} c cc cc c @{}}", r"\toprule",
     r"\textbf{Transformation} & $N$ & \textbf{Pearson} $r$ & \textbf{Naive} $p$ & \textbf{HAC} $t$ [$p$] & \textbf{Eff.} $N$ [$p$] & \textbf{Spearman} $\rho$ \\", r"\midrule"]
for _, r_ in T.iterrows():
    L.append(f"{r_.measure} & {int(r_.N)} & {f2(r_.pearson)} & {r_.p_naive:.3f} & {f2(r_.t_hac, 2)} [{r_.p_hac:.3f}] & {r_.n_eff:.0f} [{r_.p_neff:.3f}] & {f2(r_.spearman, 2)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "T_sfc.tex").write_text("\n".join(L), encoding="utf-8")
