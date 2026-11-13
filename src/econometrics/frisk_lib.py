"""
frisk_lib.py -- shared routines for the revised empirical application (V2).

Design principle: inference rests on the TIME dimension (T = 120-132 months per country),
not on the cross-section (G = 3 countries).  All covariance matrices are
heteroskedasticity-and-autocorrelation-consistent (Newey-West / Bartlett kernel) and, when
several countries or several horizons are estimated together, are computed JOINTLY from the
stacked score vectors, so that cross-country correlation (Driscoll-Kraay, 1998; SUR-type
system) and cross-horizon correlation (local projections) are both accounted for.

Everything here uses only numpy / pandas / scipy.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]          # project root (folder with the databases)
OUT = ROOT / "results"
DATA = ROOT / "data" / "derived"

COUNTRIES = ["Chile", "Mexico", "Colombia"]
GRID = pd.date_range("2014-01-01", "2024-12-01", freq="MS")   # 132 months
T_GRID = len(GRID)
TYPOLOGIES = {
    "General Fraud Complaints": "std_ln_FRISK_General_Fraud_Complaints",
    "Social Engineering": "std_ln_FRISK_Social_Engineering",
    "Identity Theft": "std_ln_FRISK_Identity_Theft",
    "Payment Fraud": "std_ln_FRISK_Payment_Fraud",
    "Account Takeover": "std_ln_FRISK_Account_Takeover",
}
# window in which Colombia's pooled IFI falls smoothly by ~43% and in which the raw SFC account
# counts show breaks (see data note in the paper); used only in a robustness check
COL_BREAK = ("2022-07-01", "2024-01-01")
COVID = ("2020-03-01", "2020-12-01")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_panel():
    """Monthly country panel on the full 2014-01..2024-12 grid (NaN where unavailable)."""
    ifi_p = pd.read_csv(ROOT / "data" / "public" / "ifi" / "IFI_Final_pooled.csv")
    ifi_w = pd.read_csv(ROOT / "data" / "public" / "ifi" / "IFI_Final.csv", encoding="utf-8-sig").rename(columns={"IFI": "ifi_w"})
    ctr = pd.read_csv(ROOT / "data" / "public" / "macro" / "Controles_Final.csv", encoding="utf-8-sig")
    fr = pd.read_csv(ROOT / "data" / "public" / "frisk" / "dataset_FRISK_panel_clean.csv")
    fred = pd.read_csv(ROOT / "data" / "public" / "macro" / "chile_cpi_fred.csv", parse_dates=["observation_date"])

    for d in (ifi_p, ifi_w, ctr, fr):
        d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=1))
    ctr = ctr.rename(columns={"economy_index_2015": "act", "unemployment_rate": "unemp",
                              "inflation_index": "cpi_raw", "conexiones_moviles": "mobile"})

    # --- CPI: the compiled control file stores the SAME price index for Chile and Colombia.
    # The original paper replaced Chile's series by FRED/OECD (2015=100, to Dec-2023) chained
    # with the 2024 monthly growth of the file's series.  Colombia's series is NOT a Colombian CPI
    # and is therefore NOT used as a control anywhere in the revised analysis.
    chl = ctr[ctr.country == "Chile"].merge(fred.rename(columns={"observation_date": "date", "CHLCPIALLMINMEI": "fred"}),
                                            on="date", how="left")
    d23 = chl.loc[chl.date == "2023-12-01", "cpi_raw"].iloc[0]
    f23 = chl.loc[chl.date == "2023-12-01", "fred"].iloc[0]
    chl["cpi"] = np.where(chl.date <= "2023-12-01", chl.fred, f23 * chl.cpi_raw / d23)
    ctr = ctr.merge(chl[["date", "cpi"]].assign(country="Chile"), on=["date", "country"], how="left")
    ctr["cpi_legacy"] = np.where(ctr.country == "Chile", ctr.cpi, ctr.cpi_raw)   # used ONLY to replicate the old TWFE

    # Colombia: replace the duplicated series by the DANE CPI rebuilt from the monthly changes (2013-2022; see build_ipc_colombia.py)
    ipc = pd.read_csv(ROOT / "data" / "public" / "macro" / "ipc_colombia_monthly.csv")
    ipc["date"] = pd.to_datetime(dict(year=ipc.year, month=ipc.month, day=1))
    ctr = ctr.merge(ipc[["date", "cpi_idx"]].assign(country="Colombia"), on=["date", "country"], how="left")
    ctr["cpi"] = np.where(ctr.country == "Colombia", ctr.cpi_idx, np.where(ctr.country == "Chile", ctr.cpi, ctr.cpi_raw))
    panel = (ctr[["country", "date", "act", "unemp", "cpi", "cpi_legacy", "mobile"]]
             .merge(ifi_p[["country", "date", "IFI"]].rename(columns={"IFI": "ifi"}), on=["country", "date"], how="left")
             .merge(ifi_w[["country", "date", "ifi_w"]], on=["country", "date"], how="left")
             .merge(fr[["country", "date", "fraud_cases_total", "std_ln_FRISK_total"] + list(TYPOLOGIES.values())],
                    on=["country", "date"], how="left"))
    panel = panel.rename(columns={"std_ln_FRISK_total": "r", "ifi": "ifi_pooled"})
    new = pd.read_csv(DATA / "ifi_rebuilt.csv", parse_dates=["date"])[["country", "date", "ifi_new"]].rename(columns={"ifi_new": "ifi"})
    panel = panel.merge(new, on=["country", "date"], how="left")
    out = {}
    for c in COUNTRIES:
        g = panel[panel.country == c].set_index("date").reindex(GRID)
        g["country"] = c
        out[c] = g
    return out


# --------------------------------------------------------------------------------------
# HAC / Driscoll-Kraay covariance for a stack of OLS models
# --------------------------------------------------------------------------------------
def hac_joint(models, L, small_sample=True):
    """
    Joint covariance of the stacked OLS coefficients of several regressions.

    models : list of dict(X=(n,k) array, u=(n,) residuals, idx=(n,) integer positions on the
             monthly grid).  Rows of different models that share a date are linked through
             their scores (this is what delivers cross-country / cross-horizon covariances).
             If several rows of ONE model share a date (pooled panel), their scores are summed
             by date: this is the Driscoll-Kraay (1998) estimator.
    L      : Bartlett bandwidth (autocovariances 1..L).  L = 0 gives White/EHW (HC) errors
             only if no two rows share a date.
    Returns (V, slices) where slices[m] is the slice of the stacked coefficient vector.
    """
    k_tot = sum(m["X"].shape[1] for m in models)
    S = np.zeros((T_GRID, k_tot))
    B = np.zeros((k_tot, k_tot))
    slices, o = [], 0
    for m in models:
        X, u, idx = m["X"], m["u"], np.asarray(m["idx"])
        n, k = X.shape
        corr = np.sqrt(n / (n - k)) if small_sample else 1.0
        sc = X * (u * corr)[:, None]
        np.add.at(S[:, o:o + k], idx, sc)       # accumulate scores by date (pooled rows -> Driscoll-Kraay)
        B[o:o + k, o:o + k] = np.linalg.inv(X.T @ X)
        slices.append(slice(o, o + k))
        o += k
    Om = S.T @ S
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = S[l:].T @ S[:-l]
        Om += w * (G + G.T)
    return B @ Om @ B, slices


def ols(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, y - X @ b


def wald_F(b, V, R=None, q_df=None, n_k=None):
    """Wald test of R b = 0.  Returns (W, q, p_chi2, p_F)."""
    b = np.asarray(b)
    if R is None:
        R = np.eye(len(b))
    r = R @ b
    Vr = R @ V @ R.T
    W = float(r @ np.linalg.solve(Vr, r))
    q = R.shape[0]
    p_chi = 1 - stats.chi2.cdf(W, q)
    p_F = 1 - stats.f.cdf(W / q, q, n_k) if n_k else np.nan
    return W, q, p_chi, p_F


# --------------------------------------------------------------------------------------
# Local-projection design
# --------------------------------------------------------------------------------------
def make_design(g, *, p=3, H=6, y="ifi", x="r", lagy=None, season=True, controls=(), x_mode="diff",
                trend=False, extra_lag=False, drop_windows=(), x_smooth=None, trim_sd=None):
    """
    Design matrices for  y_{t+h} - y_{t-1} = a_h + b_h * shock_t + lags + (controls) + (season) + u.

    y, x      : names of the dependent variable (levels) and of the shock variable.
    lagy      : variable whose lagged first differences enter as controls (default: y itself).
    x_mode    : 'diff'  -> shock_t = x_t - x_{t-1}; lags of dLagy and dX              (baseline)
                'level' -> shock_t = x_t;           lags of dLagy and of x in levels  (levels / lag-augmented check)
    extra_lag : add lag p+1 of every lagged regressor (lag augmentation; Montiel Olea & Plagborg-Moller 2021).
    x_smooth  : int m -> x is replaced by its m-month trailing moving average, rescaled to the original std. dev.
    trim_sd   : float -> months in which |d y| exceeds trim_sd standard deviations are set to missing.
    drop_windows : list of (start, end) strings in which y is set to missing.
    The estimation sample is COMMON to all horizons 0..H (rows with y_{t+H} observed).
    """
    g = g.copy()
    lagy = lagy or y
    for a, b in drop_windows:
        g.loc[a:b, y] = np.nan
    if trim_sd:
        d = g[y].diff()
        z = (d - d.mean()) / d.std()
        g.loc[z.abs() > trim_sd, y] = np.nan
    if x_smooth:
        sd0 = g[x].std()
        sm = g[x].rolling(x_smooth).mean()
        g[x] = sm * (sd0 / sm.std())
    yv, xv = g[y], g[x]
    dly = g[lagy].diff()
    dx = xv.diff()
    cols, data = [], {}
    data["shock"] = dx if x_mode == "diff" else xv
    cols.append("shock")
    nl = p + (1 if extra_lag else 0)
    for j in range(1, nl + 1):
        data[f"dy_l{j}"] = dly.shift(j); cols.append(f"dy_l{j}")
        data[f"x_l{j}"] = (dx.shift(j) if x_mode == "diff" else xv.shift(j)); cols.append(f"x_l{j}")
    for z in controls:                           # first differences of controls: contemporaneous + p lags
        dz = np.log(g[z]).diff() * 100 if z in ("act", "cpi") else g[z].diff()
        data[f"{z}_l0"] = dz; cols.append(f"{z}_l0")
        for j in range(1, p + 1):
            data[f"{z}_l{j}"] = dz.shift(j); cols.append(f"{z}_l{j}")
    X = pd.DataFrame(data, index=g.index)
    if season:
        for m in range(2, 13):
            X[f"m{m}"] = (g.index.month == m).astype(float)
    if trend:
        X["trend"] = (np.arange(len(g)) - len(g) / 2) / 12.0
    Y = pd.DataFrame({h: yv.shift(-h) - yv.shift(1) for h in range(0, H + 1)}, index=g.index)
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    sdy = float(yv.diff()[ok].std())              # monthly s.d. of the change in y on the estimation sample
    X, Y = X[ok], Y[ok]
    X.insert(0, "const", 1.0)
    idx = np.array([GRID.get_loc(d) for d in X.index])
    return dict(X=X.values, Y=Y.values, idx=idx, cols=list(X.columns), n=len(X), dates=X.index, sdy=sdy)


def fit_lp(des, L, H=6, small_sample=True):
    """OLS for every horizon h=0..H on the common sample; returns betas on 'shock' with joint HAC covariance."""
    X, Y = des["X"], des["Y"]
    ib = des["cols"].index("shock")
    models, betas = [], []
    for h in range(H + 1):
        b, u = ols(X, Y[:, h])
        betas.append(b[ib])
        models.append(dict(X=X, u=u, idx=des["idx"]))
    V, sl = hac_joint(models, L, small_sample)
    pos = [s.start + ib for s in sl]
    return np.array(betas), V[np.ix_(pos, pos)]


def sup_t_crit(V, level=0.95, ndraw=200_000, seed=20260101):
    """Simultaneous (sup-t) critical value of Montiel Olea & Plagborg-Moller (2019)."""
    se = np.sqrt(np.diag(V))
    R = V / np.outer(se, se)
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(len(se)), R, size=ndraw, method="cholesky")
    return float(np.quantile(np.abs(z).max(axis=1), level))


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
