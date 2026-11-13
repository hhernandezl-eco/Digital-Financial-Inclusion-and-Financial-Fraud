"""
frisk_est.py -- estimators used in Section 4.5 of the revised paper (V2).
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, coint
from frisk_lib import *

warnings.filterwarnings("ignore")

P = 3          # baseline number of lags of dIFI and dFRISK
H = 6          # maximum horizon (months)
L_LP = 8       # Bartlett bandwidth, local projections (H + 2)
L_FD = 4       # Newey-West rule floor(4 (T/100)^(2/9)) for one-step-ahead regressions
HS = np.arange(H + 1)


# ======================================================================================
# 1. Stationarity diagnostics
# ======================================================================================
def stationarity(panel):
    rows = []
    for c in COUNTRIES:
        g = panel[c]
        for name, var in (("IFI", "ifi"), ("FRISK", "r")):
            s = g[var].dropna()
            a_ct = adfuller(s, regression="ct", autolag="AIC", maxlag=12)
            a_c = adfuller(s, regression="c", autolag="AIC", maxlag=12)
            k_ct = kpss(s, regression="ct", nlags="auto")
            ds = s.diff().dropna()
            a_d = adfuller(ds, regression="c", autolag="AIC", maxlag=12)
            k_d = kpss(ds, regression="c", nlags="auto")
            rows.append(dict(country=c, series=name, T=len(s),
                             adf_ct=a_ct[0], adf_ct_p=a_ct[1], adf_c=a_c[0], adf_c_p=a_c[1], kpss_ct=k_ct[0],
                             d_adf=a_d[0], d_adf_p=a_d[1], d_kpss=k_d[0],
                             d_acf1=float(pd.Series(ds).autocorr(1))))
    st = pd.DataFrame(rows)
    eg = []
    for c in COUNTRIES:
        g = panel[c][["ifi", "r"]].dropna()
        t, p, _ = coint(g["ifi"], g["r"], trend="ct", autolag="aic")
        eg.append(dict(country=c, T=len(g), eg_stat=t, eg_p=p))
    return st, pd.DataFrame(eg)


# ======================================================================================
# 2. First-difference distributed-lag regressions (pooled with Driscoll-Kraay; by country with HAC)
# ======================================================================================
def _fd_frame(g, K, p, y="ifi", x="r"):
    dy, dx = g[y].diff(), g[x].diff()
    data = {f"dx_l{k}": dx.shift(k) for k in range(K + 1)}
    for j in range(1, p + 1):
        data[f"dy_l{j}"] = dy.shift(j)
    X = pd.DataFrame(data, index=g.index)
    for m in range(2, 13):
        X[f"m{m}"] = (g.index.month == m).astype(float)
    ok = X.notna().all(axis=1) & dy.notna()
    return X[ok], dy[ok]


def fd_dl(panel, K=6, p=P, L=L_FD, y="ifi", x="r"):
    """Country-by-country FD distributed-lag regressions (system, joint HAC) + pooled DK regression."""
    frames = {c: _fd_frame(panel[c], K, p, y, x) for c in COUNTRIES}
    models, fits = [], {}
    for c in COUNTRIES:
        X, yy = frames[c]
        Xc = np.column_stack([np.ones(len(X)), X.values])
        b, u = ols(Xc, yy.values)
        cols = ["const"] + list(X.columns)
        idx = np.array([GRID.get_loc(d) for d in X.index])
        models.append(dict(X=Xc, u=u, idx=idx))
        fits[c] = dict(b=b, cols=cols, n=len(X), sdy=float(yy.std()))
    V, sl = hac_joint(models, L)
    res = {}
    for i, c in enumerate(COUNTRIES):
        f = fits[c]
        cols = f["cols"]
        pos = [sl[i].start + cols.index(f"dx_l{k}") for k in range(K + 1)]
        res[c] = dict(theta=f["b"][[cols.index(f"dx_l{k}") for k in range(K + 1)]],
                      V=V[np.ix_(pos, pos)], pos=pos, n=f["n"], k=len(cols), sdy=f["sdy"])
    # ---- pooled: common theta_k, common seasonals, country-specific constants and AR terms --------
    rows_X, rows_y, rows_idx = [], [], []
    for ci, c in enumerate(COUNTRIES):
        X, yy = frames[c]
        n = len(X)
        common = X[[f"dx_l{k}" for k in range(K + 1)] + [f"m{m}" for m in range(2, 13)]].values
        spec = np.zeros((n, 1 + p))
        spec[:, 0] = 1.0
        spec[:, 1:] = X[[f"dy_l{j}" for j in range(1, p + 1)]].values
        blocks = np.zeros((n, 3 * (1 + p)))
        blocks[:, ci * (1 + p):(ci + 1) * (1 + p)] = spec
        rows_X.append(np.column_stack([common, blocks]))
        rows_y.append(yy.values)
        rows_idx.append(np.array([GRID.get_loc(d) for d in X.index]))
    Xp, yp, ip = np.vstack(rows_X), np.concatenate(rows_y), np.concatenate(rows_idx)
    b, u = ols(Xp, yp)
    Vp, _ = hac_joint([dict(X=Xp, u=u, idx=ip)], L)           # same-date rows summed -> Driscoll-Kraay
    pos = list(range(K + 1))
    res["Pooled"] = dict(theta=b[pos], V=Vp[np.ix_(pos, pos)], pos=pos, n=len(yp), k=Xp.shape[1], sdy=float(yp.std()))
    return res, V, sl


# ======================================================================================
# 3. Granger-causality (Wald-HAC) country by country and jointly
# ======================================================================================
def granger(panel, cause, effect, p, L=L_FD):
    models, info = [], []
    for c in COUNTRIES:
        g = panel[c]
        de, dc = g[effect].diff(), g[cause].diff()
        data = {}
        for j in range(1, p + 1):
            data[f"e_l{j}"] = de.shift(j)
            data[f"c_l{j}"] = dc.shift(j)
        X = pd.DataFrame(data, index=g.index)
        for m in range(2, 13):
            X[f"m{m}"] = (g.index.month == m).astype(float)
        ok = X.notna().all(axis=1) & de.notna()
        X, yy = X[ok], de[ok]
        Xc = np.column_stack([np.ones(len(X)), X.values])
        b, u = ols(Xc, yy.values)
        cols = ["const"] + list(X.columns)
        models.append(dict(X=Xc, u=u, idx=np.array([GRID.get_loc(d) for d in X.index])))
        info.append(dict(b=b, cols=cols, n=len(X)))
    V, sl = hac_joint(models, L)
    out = {}
    allpos = []
    for i, c in enumerate(COUNTRIES):
        cols = info[i]["cols"]
        pos = [sl[i].start + cols.index(f"c_l{j}") for j in range(1, p + 1)]
        bsub = np.concatenate([info[i]["b"][[cols.index(f"c_l{j}") for j in range(1, p + 1)]]])
        W, q, pchi, pF = wald_F(bsub, V[np.ix_(pos, pos)], n_k=info[i]["n"] - len(cols))
        out[c] = dict(W=W, q=q, p_chi=pchi, p_F=pF, n=info[i]["n"])
        allpos += pos
    ball = np.concatenate([info[i]["b"][[info[i]["cols"].index(f"c_l{j}") for j in range(1, p + 1)]] for i in range(3)])
    W, q, pchi, pF = wald_F(ball, V[np.ix_(allpos, allpos)],
                            n_k=min(inf["n"] - len(inf["cols"]) for inf in info))
    out["Joint (3 countries)"] = dict(W=W, q=q, p_chi=pchi, p_F=pF, n=sum(inf["n"] for inf in info))
    return out


# ======================================================================================
# 4. Local projections
# ======================================================================================
def lp_system(panel, spec=None, per_country=None, L=L_LP):
    """
    Country-by-country LPs for h=0..H estimated as ONE system with joint HAC covariance.
    Returns dict(beta (3,H+1), V (3(H+1), 3(H+1)), sdy (3,), n (3,)).
    """
    spec = dict(spec or {})
    per_country = per_country or {}
    models, beta, sdy, nn, kk, pos = [], np.zeros((3, H + 1)), np.zeros(3), np.zeros(3, int), np.zeros(3, int), []
    for ci, c in enumerate(COUNTRIES):
        kw = {**spec, **per_country.get(c, {})}
        d = make_design(panel[c], p=kw.pop("p", P), H=H, **kw)
        ib = d["cols"].index("shock")
        for h in HS:
            b, u = ols(d["X"], d["Y"][:, h])
            beta[ci, h] = b[ib]
            models.append(dict(X=d["X"], u=u, idx=d["idx"]))
        sdy[ci], nn[ci], kk[ci] = d["sdy"], d["n"], d["X"].shape[1]
        pos += [(ci * (H + 1) + h, ib) for h in HS]
    V, sl = hac_joint(models, L)
    ix = [sl[m].start + ib for m, ib in pos]
    return dict(beta=beta, V=V[np.ix_(ix, ix)], sdy=sdy, n=nn, k=kk)


def lp_summary(res):
    """Average response over h=1..H, cumulative response at H, MG average, homogeneity, joint tests."""
    beta, V, sdy, n, kk = res["beta"], res["V"], res["sdy"], res["n"], res["k"]
    nb = 3 * (H + 1)
    out = {}

    def wvec(ci, hs, scale=1.0):
        w = np.zeros(nb)
        for h in hs:
            w[ci * (H + 1) + h] = scale / len(hs)
        return w

    bvec = beta.reshape(-1)
    for ci, c in enumerate(COUNTRIES):
        for tag, hs in (("avg", range(1, H + 1)), ("h6", [H]), ("h3", [3])):
            w = wvec(ci, hs)
            est, se = float(w @ bvec), float(np.sqrt(w @ V @ w))
            out[(c, tag)] = dict(est=est, se=se, est_sd=est / sdy[ci], se_sd=se / sdy[ci],
                                 t=est / se, p=2 * (1 - stats.norm.cdf(abs(est / se))))
        # joint Wald for H0: beta_1 = ... = beta_H = 0 (country c)
        pos = [ci * (H + 1) + h for h in range(1, H + 1)]
        W, q, pchi, pF = wald_F(beta[ci, 1:], V[np.ix_(pos, pos)], n_k=int(n[ci] - kk[ci]))
        out[(c, "joint")] = dict(W=W, q=q, p_chi=pchi, p_F=pF)
    # mean-group (equal weights) in units of each country's s.d. of monthly dIFI
    for tag, hs in (("avg", range(1, H + 1)), ("h6", [H]), ("h3", [3])):
        w = sum(wvec(ci, hs, 1.0 / sdy[ci]) for ci in range(3)) / 3.0
        est, se = float(w @ bvec), float(np.sqrt(w @ V @ w))
        out[("MG", tag)] = dict(est_sd=est, se_sd=se, t=est / se, p=2 * (1 - stats.norm.cdf(abs(est / se))))
        # MG in IFI points (pooled scale)
        w2 = sum(wvec(ci, hs, 1.0) for ci in range(3)) / 3.0
        est2, se2 = float(w2 @ bvec), float(np.sqrt(w2 @ V @ w2))
        out[("MG_pts", tag)] = dict(est=est2, se=se2, t=est2 / se2, p=2 * (1 - stats.norm.cdf(abs(est2 / se2))))
        # homogeneity across countries (in s.d. units)
        R = np.vstack([wvec(0, hs, 1 / sdy[0]) - wvec(1, hs, 1 / sdy[1]),
                       wvec(1, hs, 1 / sdy[1]) - wvec(2, hs, 1 / sdy[2])])
        W, q, pchi, pF = wald_F(R @ bvec, R @ V @ R.T, R=np.eye(2), n_k=int(min(n - kk)))
        out[("hom", tag)] = dict(W=W, q=q, p_chi=pchi, p_F=pF)
    return out


def sup_bands(res, ci):
    """pointwise and sup-t critical values for country ci (all horizons 0..H)."""
    pos = [ci * (H + 1) + h for h in HS]
    Vc = res["V"][np.ix_(pos, pos)]
    return Vc, sup_t_crit(Vc)


# ---------------------------------- block bootstrap ----------------------------------
def lp_block_bootstrap(panel, spec=None, B=2000, block=12, seed=20260101):
    """Moving-block bootstrap on calendar time (same blocks in every country) of the LP rows."""
    spec = dict(spec or {})
    rng = np.random.default_rng(seed)
    des = []
    for c in COUNTRIES:
        kw = dict(spec)
        d = make_design(panel[c], p=kw.pop("p", P), H=H, **kw)
        row_of_pos = -np.ones(T_GRID, int)
        row_of_pos[d["idx"]] = np.arange(d["n"])
        des.append((d, row_of_pos))
    base = np.array([np.linalg.lstsq(d["X"], d["Y"], rcond=None)[0][1] for d, _ in des])       # (3, H+1)
    sdy = np.array([d["sdy"] for d, _ in des])
    nblk = int(np.ceil(T_GRID / block))
    draws = np.zeros((B, 3, H + 1))
    for b in range(B):
        starts = rng.integers(0, T_GRID - block + 1, size=nblk)
        pos = np.concatenate([np.arange(s, s + block) for s in starts])[:T_GRID]
        for ci, (d, rop) in enumerate(des):
            sel = rop[pos]
            sel = sel[sel >= 0]
            coef = np.linalg.lstsq(d["X"][sel], d["Y"][sel], rcond=None)[0]
            draws[b, ci] = coef[1]
    return base, draws, sdy


# ======================================================================================
# 5. Legacy levels regression (original Table 8) and its diagnostics
# ======================================================================================
def legacy_levels(panel):
    from linearmodels.panel import PanelOLS
    rows = []
    d = pd.concat([panel[c].assign(country=c) for c in COUNTRIES]).reset_index().rename(columns={"index": "date"})
    d = d.sort_values(["country", "date"])
    ctrl = ["unemp", "mobile", "act"]

    def dw(res):
        e = res.resids.unstack(0) if False else res.resids
        e = e.reset_index()
        e.columns = ["country", "date", "e"]
        num = den = 0.0
        for _, gg in e.sort_values(["country", "date"]).groupby("country"):
            v = gg.e.values
            num += np.sum(np.diff(v) ** 2)
            den += np.sum(v ** 2)
        return num / den

    def run(label, depvar, extra_trend=False, cov="clustered", L=12, diff=False):
        dd = d.dropna(subset=[depvar, "r"] + ctrl).copy()
        if extra_trend:
            tt = (dd.date.dt.year - 2014) * 12 + dd.date.dt.month
            for c in ["Colombia", "Mexico"]:
                dd[f"trend_{c}"] = np.where(dd.country == c, tt, 0.0)
        dd = dd.set_index(["country", "date"])
        X = dd[["r"] + ctrl + ([c for c in dd.columns if c.startswith("trend_")] if extra_trend else [])]
        mod = PanelOLS(dd[depvar], X, entity_effects=True, time_effects=True)
        if cov == "clustered":
            r = mod.fit(cov_type="clustered", cluster_entity=True)
        else:
            r = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=L)
        rows.append(dict(spec=label, b=r.params["r"], se=r.std_errors["r"], p=r.pvalues["r"], n=int(r.nobs), dw=dw(r)))

    run("(a) TWFE levels, pooled IFI, country-clustered s.e. (original Table 8)", "ifi_pooled")
    run("(b) same, Driscoll-Kraay s.e. (bandwidth 12)", "ifi_pooled", cov="kernel", L=12)
    run("(c) same as (a), IFI normalized within country", "ifi")
    run("(d) same as (b), IFI normalized within country", "ifi", cov="kernel", L=12)
    run("(e) same as (a), plus country-specific linear trends", "ifi_pooled", extra_trend=True)
    # first differences, country FE, DK
    dd = d.copy()
    for v in ("ifi_pooled", "ifi", "r"):
        dd["d_" + v] = dd.groupby("country")[v].diff()
    for v, lab in (("d_ifi_pooled", "(f) first differences, pooled IFI, country FE, Driscoll-Kraay s.e."),
                   ("d_ifi", "(g) first differences, within-normalized IFI, country FE, Driscoll-Kraay s.e.")):
        x = dd.dropna(subset=[v, "d_r"]).set_index(["country", "date"])
        r = PanelOLS(x[v], x[["d_r"]], entity_effects=True).fit(cov_type="kernel", kernel="bartlett", bandwidth=12)
        e = r.resids.reset_index()
        e.columns = ["country", "date", "e"]
        num = den = 0.0
        for _, gg in e.sort_values(["country", "date"]).groupby("country"):
            v_ = gg.e.values
            num += np.sum(np.diff(v_) ** 2); den += np.sum(v_ ** 2)
        rows.append(dict(spec=lab, b=r.params["d_r"], se=r.std_errors["d_r"], p=r.pvalues["d_r"], n=int(r.nobs), dw=num / den))
    return pd.DataFrame(rows)


# ======================================================================================
# 3b. Granger causality with a wild-bootstrap p-value (small-sample valid, heteroskedasticity robust)
# ======================================================================================
def _hc_wald(X, y, cols_idx):
    """HC1-robust Wald statistic for the coefficients in cols_idx."""
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    meat = (X * u[:, None]).T @ (X * u[:, None]) * n / (n - k)
    V = XtXi @ meat @ XtXi
    bs = b[cols_idx]
    return float(bs @ np.linalg.solve(V[np.ix_(cols_idx, cols_idx)], bs))


def granger_wild(panel, cause, effect, p, B=999, seed=20260103):
    """
    Wild-bootstrap Granger-causality test (H0: lags 1..p of d(cause) do not enter the d(effect) equation).
    Statistic: HC1 Wald, per country and summed over countries (joint).  Bootstrap: restricted residuals multiplied
    by Rademacher weights drawn per calendar month and COMMON to the three countries (keeps cross-country dependence).
    """
    rng = np.random.default_rng(seed)
    data = []
    for c in COUNTRIES:
        g = panel[c]
        de, dc = g[effect].diff(), g[cause].diff()
        d = {}
        for j in range(1, p + 1):
            d[f"e_l{j}"] = de.shift(j); d[f"c_l{j}"] = dc.shift(j)
        X = pd.DataFrame(d, index=g.index)
        for m in range(2, 13):
            X[f"m{m}"] = (g.index.month == m).astype(float)
        ok = X.notna().all(axis=1) & de.notna()
        X, y = X[ok], de[ok]
        Xc = np.column_stack([np.ones(len(X)), X.values])
        cols = ["const"] + list(X.columns)
        ci = [cols.index(f"c_l{j}") for j in range(1, p + 1)]
        keep = [i for i in range(Xc.shape[1]) if i not in ci]
        Xr = Xc[:, keep]
        br = np.linalg.lstsq(Xr, y.values, rcond=None)[0]
        ur = y.values - Xr @ br
        data.append(dict(X=Xc, y=y.values, ci=ci, Xr=Xr, br=br, ur=ur, idx=np.array([GRID.get_loc(t) for t in X.index])))
    W0 = np.array([_hc_wald(d["X"], d["y"], d["ci"]) for d in data])
    Wb = np.zeros((B, 3))
    for b in range(B):
        v = rng.choice([-1.0, 1.0], size=T_GRID)
        for i, d in enumerate(data):
            ystar = d["Xr"] @ d["br"] + v[d["idx"]] * d["ur"]
            Wb[b, i] = _hc_wald(d["X"], ystar, d["ci"])
    out = {}
    for i, c in enumerate(COUNTRIES):
        out[c] = dict(W=W0[i], q=p, p=(1 + np.sum(Wb[:, i] >= W0[i])) / (B + 1))
    out["Joint (3 countries)"] = dict(W=W0.sum(), q=3 * p, p=(1 + np.sum(Wb.sum(1) >= W0.sum())) / (B + 1))
    return out


def levels_controls_check(panel):
    """Levels TWFE (country-clustered s.e.) under alternative control sets: does the instability depend on the controls?"""
    from linearmodels.panel import PanelOLS
    d = pd.concat([panel[c].assign(country=c) for c in COUNTRIES]).reset_index().rename(columns={"index": "date"})
    d = d.sort_values(["country", "date"])
    rows = []
    for lab, ctr in (("no controls", []), ("unemployment, mobile, activity (no prices)", ["unemp", "mobile", "act"]),
                     ("unemployment, activity", ["unemp", "act"])):
        for dep, dl in (("ifi_pooled", "pooled IFI"), ("ifi", "within-normalized IFI")):
            for trend in (False, True):
                dd = d.dropna(subset=[dep, "r"] + ctr).copy()
                if trend:
                    tt = (dd.date.dt.year - 2014) * 12 + dd.date.dt.month
                    for c in ("Colombia", "Mexico"):
                        dd[f"tr_{c}"] = np.where(dd.country == c, tt, 0.0)
                dd = dd.set_index(["country", "date"])
                X = dd[["r"] + ctr + ([k for k in dd.columns if k.startswith("tr_")] if trend else [])]
                r = PanelOLS(dd[dep], X, entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
                rows.append(dict(controls=lab, dep=dl, trend=trend, b=float(r.params["r"]), se=float(r.std_errors["r"]), n=int(r.nobs)))
    return pd.DataFrame(rows)
