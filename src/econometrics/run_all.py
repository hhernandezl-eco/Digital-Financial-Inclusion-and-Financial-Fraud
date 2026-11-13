"""
run_all.py -- runs every estimate reported in Section 4.5 and Appendix A of the revised paper (V2)
and stores the results in results/results.pkl (+ csv files).

Run from the folder  code :   python run_all.py
All specification choices (p = 3 lags, H = 6 months, Bartlett bandwidth 8 for local projections, 4 for
one-step regressions, 12-month block bootstrap) were fixed BEFORE looking at the estimates.
"""
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from frisk_est import *

OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
panel = load_panel()
R = {}

# --- save the panel actually used -------------------------------------------------------------
flat = pd.concat([panel[c].assign(country=c) for c in COUNTRIES]).reset_index().rename(columns={"index": "date"})
flat.to_csv(DATA / "panel_monthly.csv", index=False)

# 1. stationarity ---------------------------------------------------------------------------------
R["unit_root"], R["eg"] = stationarity(panel)
R["unit_root"].to_csv(OUT / "unit_root.csv", index=False)

# 2. first-difference distributed-lag regressions ------------------------------------------------
fd, Vfd, slfd = fd_dl(panel)
R["fd"] = fd
# cross-equation tests from the joint (system) covariance of the three country regressions
cum_w = {}
nb = Vfd.shape[0]
for i, c in enumerate(COUNTRIES):
    w = np.zeros(nb)
    for p_ in fd[c]["pos"]:
        w[p_] = 1.0
    cum_w[c] = w
bfull = np.zeros(nb)
for i, c in enumerate(COUNTRIES):
    for k, p_ in enumerate(fd[c]["pos"]):
        bfull[p_] = fd[c]["theta"][k]
Rm = np.vstack([cum_w["Chile"] - cum_w["Mexico"], cum_w["Mexico"] - cum_w["Colombia"]])
R["fd_hom"] = wald_F(Rm @ bfull, Rm @ Vfd @ Rm.T, R=np.eye(2), n_k=min(fd[c]["n"] - fd[c]["k"] for c in COUNTRIES))
wmg = (cum_w["Chile"] + cum_w["Mexico"] + cum_w["Colombia"]) / 3
R["fd_mg"] = (float(wmg @ bfull), float(np.sqrt(wmg @ Vfd @ wmg)))

# 3. Granger causality (wild-bootstrap p-values; HAC-Wald kept only as a by-product) --------------
R["granger"] = {}
for p_ in (3, 6):
    R["granger"][("FRISK->IFI", p_)] = granger_wild(panel, "r", "ifi", p_, B=999)
    R["granger"][("IFI->FRISK", p_)] = granger_wild(panel, "ifi", "r", p_, B=999)
    R["granger"][("FRISK->IFI", p_, "hac")] = granger(panel, "r", "ifi", p_)      # not reported (oversized in MC)

def _variant(mod):
    return {c: mod(panel[c].copy(), c) for c in COUNTRIES}

def _trim(g, c):
    d = g.ifi.diff(); z = (d - d.mean()) / d.std(); g = g.copy(); g.loc[z.abs() > 3, "ifi"] = np.nan; return g

gv = {
    "Baseline": lambda g, c: g,
    "Excluding Mar--Dec 2020": lambda g, c: g.assign(ifi=g.ifi.where(~((g.index >= COVID[0]) & (g.index <= COVID[1])))),
    "Outlier months trimmed": _trim,
    "Original pooled-normalized IFI": lambda g, c: g.assign(ifi=g.ifi_pooled),
}
R["granger_rob"] = {}
for name, mod in gv.items():
    pn = _variant(mod)
    for p_ in (3, 6):
        R["granger_rob"][(name, "FRISK->IFI", p_)] = granger_wild(pn, "r", "ifi", p_, B=499)
        R["granger_rob"][(name, "IFI->FRISK", p_)] = granger_wild(pn, "ifi", "r", p_, B=499)

# 4. local projections: baseline -----------------------------------------------------------------
base = lp_system(panel)
R["lp"] = base
R["lp_sum"] = lp_summary(base)
R["sup"] = {}
for ci, c in enumerate(COUNTRIES):
    Vc, crit = sup_bands(base, ci)
    R["sup"][c] = dict(V=Vc, crit=crit)

# first stage: response of FRISK itself to its own shock (VAR-style: lags of dIFI as controls)
fs_beta = np.zeros((3, H + 1)); fs_se = np.zeros((3, H + 1))
for ci, c in enumerate(COUNTRIES):
    d = make_design(panel[c], p=P, H=H, y="r", x="r", lagy="ifi")
    models = []
    for h in HS:
        b, u = ols(d["X"], d["Y"][:, h]); fs_beta[ci, h] = b[1]; models.append(dict(X=d["X"], u=u, idx=d["idx"]))
    V, sl = hac_joint(models, L_LP)
    fs_se[ci] = [np.sqrt(V[s.start + 1, s.start + 1]) for s in sl]
R["first_stage"] = (fs_beta, fs_se)

# descriptive: autocorrelation of d FRISK and sd of d IFI
R["desc"] = {c: dict(sdy=float(base["sdy"][i]), n=int(base["n"][i]), sd_dr=float(panel[c].r.diff().std()),
                     rho_dr=float(panel[c].r.diff().autocorr(1))) for i, c in enumerate(COUNTRIES)}

# 5. robustness battery --------------------------------------------------------------------------
SPECS = [
    ("Baseline ($p=3$, month dummies, HAC bandwidth 8)", dict(), {}, L_LP),
    ("One lag ($p=1$)", dict(p=1), {}, L_LP),
    ("Six lags ($p=6$)", dict(p=6), {}, L_LP),
    ("No calendar-month dummies", dict(season=False), {}, L_LP),
    ("Macro controls (change in unemployment, growth of activity)", dict(controls=("unemp", "act")), {}, L_LP),
    ("Original pooled-normalized IFI", dict(y="ifi_pooled"), {}, L_LP),
    ("FRISK as 3-month moving average", dict(x_smooth=3), {}, L_LP),
    ("Outlier months trimmed ($|\\Delta IFI|>3$ s.d.)", dict(trim_sd=3), {}, L_LP),
    ("Excluding March--December 2020", dict(drop_windows=[COVID]), {}, L_LP),
    ("Colombia: excluding July 2022--January 2024", dict(), {"Colombia": dict(drop_windows=[COL_BREAK])}, L_LP),
    ("Lag-augmented ($p+1$ lags), White s.e.", dict(extra_lag=True), {}, 0),
    ("FRISK in levels, linear trend, lag-augmented, White s.e.", dict(x_mode="level", trend=True, extra_lag=True), {}, 0),
    ("Macro controls plus inflation", dict(controls=("unemp", "act", "cpi")), {}, L_LP),
]
rob = []
for name, spec, pc, L in SPECS:
    res = lp_system(panel, spec, pc, L=L)
    sm = lp_summary(res)
    row = dict(spec=name)
    for c in COUNTRIES:
        row[f"{c}_est"], row[f"{c}_se"], row[f"{c}_p"] = sm[(c, "avg")]["est_sd"], sm[(c, "avg")]["se_sd"], sm[(c, "avg")]["p"]
        row[f"{c}_h6"], row[f"{c}_h6se"] = sm[(c, "h6")]["est_sd"], sm[(c, "h6")]["se_sd"]
    row["MG_est"], row["MG_se"], row["MG_p"] = sm[("MG", "avg")]["est_sd"], sm[("MG", "avg")]["se_sd"], sm[("MG", "avg")]["p"]
    row["hom_p"] = sm[("hom", "avg")]["p_F"]
    row["n"] = tuple(int(v) for v in res["n"])
    rob.append(row)
R["robust"] = pd.DataFrame(rob)
R["robust"].to_csv(OUT / "robustness_lp.csv", index=False)

# 6. block bootstrap (baseline) ------------------------------------------------------------------
b0, dr, sdy = lp_block_bootstrap(panel, B=2000, block=12)
def _avg(a):                    # a (..., 3, H+1) -> avg over h=1..H, in sd units
    return a[..., 1:].mean(axis=-1) / sdy
bs = dict(base=b0, draws=dr, sdy=sdy)
avg_d = _avg(dr)               # (B,3)
bs["avg_ci"] = np.percentile(avg_d, [2.5, 97.5], axis=0)
bs["mg_ci"] = np.percentile(avg_d.mean(axis=1), [2.5, 97.5])
bs["h_ci"] = np.percentile(dr, [2.5, 97.5], axis=0)           # (2,3,H+1) in IFI points
bs["h6_ci"] = np.percentile(dr[:, :, H] / sdy, [2.5, 97.5], axis=0)
R["boot"] = bs

# 7. typologies ----------------------------------------------------------------------------------
typ = []
for tname, var in TYPOLOGIES.items():
    pn = {c: panel[c].assign(rt=panel[c][var]) for c in COUNTRIES}
    res = lp_system(pn, dict(x="rt"))
    sm = lp_summary(res)
    for c in COUNTRIES:
        typ.append(dict(typology=tname, country=c, est=sm[(c, "avg")]["est_sd"], se=sm[(c, "avg")]["se_sd"],
                        t=sm[(c, "avg")]["t"], p=sm[(c, "avg")]["p"]))
    typ.append(dict(typology=tname, country="Mean group", est=sm[("MG", "avg")]["est_sd"], se=sm[("MG", "avg")]["se_sd"],
                    t=sm[("MG", "avg")]["t"], p=sm[("MG", "avg")]["p"]))
typ = pd.DataFrame(typ)
# Holm adjustment over the 15 country x typology tests
ct = typ[typ.country != "Mean group"].copy()
order = ct.p.sort_values().index
m = len(ct)
adj = {}
run_max = 0.0
for rank, ix in enumerate(order):
    val = min(1.0, (m - rank) * ct.loc[ix, "p"])
    run_max = max(run_max, val)
    adj[ix] = run_max
typ["p_holm"] = pd.Series(adj)
R["typology"] = typ
typ.to_csv(OUT / "typology_lp.csv", index=False)

# 8. legacy levels regression and diagnostics --------------------------------------------------
R["legacy"] = legacy_levels(panel)
R["legacy"].to_csv(OUT / "levels_legacy.csv", index=False)
R["levels_alt"] = levels_controls_check(panel)
R["levels_alt"].to_csv(OUT / "levels_alt_controls.csv", index=False)

with open(OUT / "results.pkl", "wb") as f:
    pickle.dump(R, f)
print("done; results saved to", OUT)
