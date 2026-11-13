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


