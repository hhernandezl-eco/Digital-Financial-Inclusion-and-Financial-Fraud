"""
mc_size.py -- Monte Carlo evidence on the size of the old and new inference procedures.

DGP (null of NO relation between fraud visibility and inclusion), calibrated to the sample:
  IFI_{c,t}  : random walk with drift, dIFI_t = mu_c + phi (dIFI_{t-1} - mu_c) + s_c u_{c,t}   (I(1))
  FRISK_{c,t}: (A) deterministic linear trend + white noise        (trend-stationary)
               (B) random walk + white noise                        (I(1) with measurement noise)
  u and the FRISK noise are correlated across countries (common component, rho = 0.3).
Procedures:
  OLD : levels two-way FE, standard errors clustered by country, t(G-1)=t(2) critical values (original Table 8)
  OLD2: levels two-way FE, Driscoll-Kraay (bandwidth 12), normal critical values
  NEW : LP system in differences, joint Newey-West/Bartlett (bandwidth 8), country-specific t-tests of the
        average response over h=1..6 and the equally weighted mean-group test (normal critical values)
Reports rejection rates at the 5% level (nominal size = 0.05).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from frisk_est import *

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(20260102)
T = len(GRID)
mu = np.array([0.0030, 0.0008, 0.0006])        # monthly drift of IFI (Chile, Mexico, Colombia), pooled scale
sg = np.array([0.0109, 0.0038, 0.0023])        # s.d. of monthly changes
phi = 0.2


def simulate(kind):
    common_u = rng.standard_normal(T)
    common_e = rng.standard_normal(T)
    ifi, fr = [], []
    for c in range(3):
        u = np.sqrt(0.3) * common_u + np.sqrt(0.7) * rng.standard_normal(T)
        d = np.zeros(T)
        for t in range(1, T):
            d[t] = mu[c] + phi * (d[t - 1] - mu[c]) + sg[c] * u[t]
        ifi.append(0.4 + np.cumsum(d))
        e = (np.sqrt(0.3) * common_e + np.sqrt(0.7) * rng.standard_normal(T)) * 0.8
        if kind == "A":
            fr.append(-0.5 + 1.8 * np.arange(T) / T + e)
        else:
            rw = np.cumsum(0.15 * rng.standard_normal(T))
            fr.append(rw + e)
    return np.array(ifi), np.array(fr)


def twfe_levels(ifi, fr, L=None):
    """Two-way FE (balanced), slope of IFI on FRISK; clustered-by-country or Driscoll-Kraay t-stat."""
    def dm(a):
        return a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean()
    y, x = dm(ifi), dm(fr)
    b = (x * y).sum() / (x * x).sum()
    u = y - b * x
    sxx = (x * x).sum()
    if L is None:                                           # cluster by country (G=3), no small-sample factor
        sc = (x * u).sum(1)
        v = (sc ** 2).sum() / sxx ** 2
        G = 3
        v *= G / (G - 1)
    else:                                                   # Driscoll-Kraay: sum scores over countries by date
        s = (x * u).sum(0)
        om = s @ s
        for l in range(1, L + 1):
            om += 2 * (1 - l / (L + 1)) * (s[l:] @ s[:-l])
        v = om / sxx ** 2
    return b / np.sqrt(v)


def to_panel(ifi, fr):
    return {c: pd.DataFrame({"ifi": ifi[i], "r": fr[i]}, index=GRID).assign(country=c) for i, c in enumerate(COUNTRIES)}


rows = []
for kind in ("A", "B"):
    rej = dict(old_t2=0, old_norm=0, old2_dk=0, new_c=[0, 0, 0], new_any=0, new_mg=0, lpj=0, fdj_c=0, fdj_pool=0, gr3_c=0, gr3_j=0, gr6_c=0, gr6_j=0)
    for r in range(NREP):
        ifi, fr = simulate(kind)
        t_old = twfe_levels(ifi, fr)
        rej["old_t2"] += abs(t_old) > stats.t.ppf(0.975, 2)
        rej["old_norm"] += abs(t_old) > 1.96
        rej["old2_dk"] += abs(twfe_levels(ifi, fr, L=12)) > 1.96
        pn = to_panel(ifi, fr)
        res = lp_system(pn)
        sm = lp_summary(res)
        anyr = 0
        for i, c in enumerate(COUNTRIES):
            rj = abs(sm[(c, "avg")]["t"]) > 1.96
            rej["new_c"][i] += rj
            anyr |= rj
        rej["new_any"] += anyr
        rej["new_mg"] += abs(sm[("MG", "avg")]["t"]) > 1.96
        for c in COUNTRIES:
            rej["lpj"] += (sm[(c, "joint")]["p_F"] < 0.05) / 3
        fd, _, _ = fd_dl(pn)
        for c in COUNTRIES:
            rej["fdj_c"] += (wald_F(fd[c]["theta"], fd[c]["V"], n_k=fd[c]["n"] - fd[c]["k"])[3] < 0.05) / 3
        rej["fdj_pool"] += wald_F(fd["Pooled"]["theta"], fd["Pooled"]["V"], n_k=fd["Pooled"]["n"] - fd["Pooled"]["k"])[3] < 0.05
        for p_, kc, kj in ((3, "gr3_c", "gr3_j"), (6, "gr6_c", "gr6_j")):
            gg = granger(pn, "r", "ifi", p_)
            for c in COUNTRIES:
                rej[kc] += (gg[c]["p_F"] < 0.05) / 3
            rej[kj] += gg["Joint (3 countries)"]["p_F"] < 0.05
    n = NREP
    rows.append(dict(dgp=kind, old_t2=rej["old_t2"] / n, old_norm=rej["old_norm"] / n, old2_dk=rej["old2_dk"] / n,
                     new_chile=rej["new_c"][0] / n, new_mexico=rej["new_c"][1] / n, new_colombia=rej["new_c"][2] / n,
                     new_any=rej["new_any"] / n, new_mg=rej["new_mg"] / n, lp_joint=rej["lpj"] / n, fd_joint_c=rej["fdj_c"] / n, fd_joint_pool=rej["fdj_pool"] / n, gr3_c=rej["gr3_c"] / n, gr3_joint=rej["gr3_j"] / n, gr6_c=rej["gr6_c"] / n, gr6_joint=rej["gr6_j"] / n, nrep=n))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "mc_size.csv", index=False)
