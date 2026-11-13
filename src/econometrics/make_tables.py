"""make_tables.py -- writes LaTeX table bodies (tabular environments) to results/T_*.tex"""
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from frisk_est import *

R = pickle.load(open(OUT / "results.pkl", "rb"))
mc = pd.read_csv(OUT / "mc_size.csv")


def f(x, d=2):
    s = f"{abs(x):.{d}f}"
    return ("$-$" if x < 0 and float(s) != 0 else "") + s


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def star(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


def zp(est, se):
    return 2 * (1 - stats.norm.cdf(abs(est / se)))


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------- 1. unit roots
ur, eg = R["unit_root"], R["eg"]
L = [r"\begin{tabular}{@{} ll c cc cc @{}}", r"\toprule",
     r" & & & \multicolumn{2}{c}{\textbf{Levels (constant + trend)}} & \multicolumn{2}{c}{\textbf{First differences}} \\",
     r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"\textbf{Country} & \textbf{Series} & $T$ & ADF $\tau$ [$p$] & KPSS $\eta$ & ADF $\tau$ [$p$] & KPSS $\eta$ \\", r"\midrule"]
for c in COUNTRIES:
    for _, r in ur[ur.country == c].iterrows():
        k1 = "\\textsuperscript{\\dag}" if r.kpss_ct > 0.146 else ""
        k2 = "\\textsuperscript{\\dag}" if r.d_kpss > 0.463 else ""
        L.append(f"{c if r.series == 'IFI' else ''} & {r.series} & {int(r['T'])} & {f(r.adf_ct)} [{r.adf_ct_p:.3f}] & {r.kpss_ct:.3f}{k1} & "
                 f"{f(r.d_adf)} [{r.d_adf_p:.3f}] & {r.d_kpss:.3f}{k2} \\\\")
    if c != COUNTRIES[-1]:
        L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}"]
write("T_unit_root.tex", "\n".join(L))

# ---------------------------------------------------------------------------- 2. FD distributed lags
fd = R["fd"]
cols = ["Pooled", "Chile", "Mexico", "Colombia"]
L = [r"\begin{tabular}{@{} l cccc @{}}", r"\toprule",
     r" & \textbf{Pooled} & \textbf{Chile} & \textbf{Mexico} & \textbf{Colombia} \\",
     r" & (Driscoll--Kraay) & (Newey--West) & (Newey--West) & (Newey--West) \\", r"\midrule"]
for k in range(7):
    lab = r"$\Delta\tilde r_{c,t}$" if k == 0 else rf"$\Delta\tilde r_{{c,t-{k}}}$"
    a, b = [], []
    for c in cols:
        th = fd[c]["theta"][k] * 1e3
        se = np.sqrt(fd[c]["V"][k, k]) * 1e3
        a.append(f"{f(th)}{star(zp(th, se))}")
        b.append(f"({se:.2f})")
    L.append(f"{lab} & " + " & ".join(a) + r" \\")
    L.append(" & " + " & ".join(b) + r" \\")
L.append(r"\midrule")
cum, cumse, jp, nn, sd = [], [], [], [], []
for c in cols:
    th, V = fd[c]["theta"], fd[c]["V"]
    s = th.sum() * 1e3
    se = np.sqrt(np.ones(7) @ V @ np.ones(7)) * 1e3
    cum.append(f"{f(s)}{star(zp(s, se))}")
    cumse.append(f"({se:.2f})")
    W, q, pchi, pF = wald_F(th, V, n_k=fd[c]["n"] - fd[c]["k"])
    jp.append(f"{pF:.3f}")
    nn.append(f"{fd[c]['n']}")
    sd.append(f"{fd[c]['sdy'] * 1e3:.2f}")
L.append(r"Cumulative effect, $\sum_{k=0}^{6}\theta_k$ & " + " & ".join(cum) + r" \\")
L.append(" & " + " & ".join(cumse) + r" \\")
L.append(r"Observations & " + " & ".join(nn) + r" \\")
L.append(r"S.d. of $\Delta IFI$ ($\times10^{3}$) & " + " & ".join(sd) + r" \\")
hom = R["fd_hom"]
L += [r"\bottomrule", r"\end{tabular}"]
write("T_fd.tex", "\n".join(L))
FD_HOM_P = hom[3]
FD_MG = R["fd_mg"]

# ---------------------------------------------------------------------------- 3. LP summary
lp, sm = R["lp"], R["lp_sum"]
beta, V, sdy, nn_, kk = lp["beta"], lp["V"], lp["sdy"], lp["n"], lp["k"]
se_all = np.sqrt(np.diag(V)).reshape(3, H + 1)
L = [r"\begin{tabular}{@{} L{5.6cm} cccc @{}}", r"\toprule",
     r" & \textbf{Chile} & \textbf{Mexico} & \textbf{Colombia} & \textbf{Mean group} \\", r"\midrule",
     r"\multicolumn{5}{@{}l}{\textit{Panel A. Response of the IFI at horizon $h$ ($\times10^{3}$ IFI points)}} \\"]
nb = 3 * (H + 1)
for h in range(H + 1):
    a, b = [], []
    for ci in range(3):
        est, se = beta[ci, h] * 1e3, se_all[ci, h] * 1e3
        a.append(f"{f(est)}{star(zp(est, se))}"); b.append(f"({se:.2f})")
    w = np.zeros(nb)
    for ci in range(3):
        w[ci * (H + 1) + h] = 1 / 3
    est = float(w @ beta.reshape(-1)) * 1e3
    se = float(np.sqrt(w @ V @ w)) * 1e3
    a.append(f"{f(est)}{star(zp(est, se))}"); b.append(f"({se:.2f})")
    L.append(f"$h={h}$ & " + " & ".join(a) + r" \\")
    L.append(" & " + " & ".join(b) + r" \\")
L.append(r"\midrule")
L.append(r"\multicolumn{5}{@{}l}{\textit{Panel B. Average response over $h=1,\dots,6$}} \\")
a, b, bc = [], [], []
for c in COUNTRIES:
    x = sm[(c, "avg")]
    a.append(f"{f(x['est'] * 1e3)}{star(x['p'])}"); b.append(f"({x['se'] * 1e3:.2f})")
x = sm[("MG_pts", "avg")]
a.append(f"{f(x['est'] * 1e3)}{star(x['p'])}"); b.append(f"({x['se'] * 1e3:.2f})")
L.append(r"IFI points ($\times10^{3}$) & " + " & ".join(a) + r" \\")
L.append(" & " + " & ".join(b) + r" \\")
a, b = [], []
for c in COUNTRIES:
    x = sm[(c, "avg")]
    a.append(f"{f(x['est_sd'])}{star(x['p'])}"); b.append(f"({x['se_sd']:.2f})")
x = sm[("MG", "avg")]
a.append(f"{f(x['est_sd'])}{star(x['p'])}"); b.append(f"({x['se_sd']:.2f})")
L.append(r"Standardized (s.d. of monthly $\Delta IFI$) & " + " & ".join(a) + r" \\")
L.append(" & " + " & ".join(b) + r" \\")
bs = R["boot"]
ci = [f"[{f(bs['avg_ci'][0, i])}, {f(bs['avg_ci'][1, i])}]" for i in range(3)] + [f"[{f(bs['mg_ci'][0])}, {f(bs['mg_ci'][1])}]"]
L.append(r"Block-bootstrap 95\% interval (standardized) & " + " & ".join(ci) + r" \\")
L.append(r"\midrule")
L.append(r"Sup-$t$ critical value (95\%, $h=0,\dots,6$) & " + " & ".join(f"{R['sup'][c]['crit']:.2f}" for c in COUNTRIES) + r" & -- \\")
L.append(r"$p$-value, equal average response across countries & \multicolumn{4}{c}{" + f"{sm[('hom', 'avg')]['p_F']:.3f}" + r"} \\")
L.append(r"S.d. of monthly $\Delta IFI$ ($\times10^{3}$) & " + " & ".join(f"{s * 1e3:.2f}" for s in sdy) + r" & -- \\")
L.append(r"Observations & " + " & ".join(str(int(n)) for n in nn_) + r" & -- \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("T_lp.tex", "\n".join(L))

# ---------------------------------------------------------------------------- 4. Granger
g = R["granger"]
L = [r"\begin{tabular}{@{} l c cccc @{}}", r"\toprule",
     r"\textbf{Null hypothesis} & \textbf{Lags} & \textbf{Chile} & \textbf{Mexico} & \textbf{Colombia} & \textbf{Joint} \\", r"\midrule"]
for direction, lab in (("FRISK->IFI", r"FRISK does not Granger-cause the IFI"), ("IFI->FRISK", r"The IFI does not Granger-cause FRISK")):
    for p_ in (3, 6):
        d = g[(direction, p_)]
        vals = [d[c]["p"] for c in COUNTRIES] + [d["Joint (3 countries)"]["p"]]
        L.append(f"{lab if p_ == 3 else ''} & {p_} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
    if direction == "FRISK->IFI":
        L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}"]
write("T_granger.tex", "\n".join(L))

# ---------------------------------------------------------------------------- A1. robustness
rob = R["robust"]
L = [r"\begin{tabular}{@{} L{5.5cm} cccc @{}}", r"\toprule",
     r"\textbf{Specification} & \textbf{Chile} & \textbf{Mexico} & \textbf{Colombia} & \textbf{Mean group} \\", r"\midrule"]
for i, r in rob.iterrows():
    cells = []
    for nm in COUNTRIES + ["MG"]:
        e, s, p = r[f"{nm}_est"], r[f"{nm}_se"], r[f"{nm}_p"]
        cells.append(f"{f(e)}{star(p)} ({s:.2f})")
    L.append(f"({i + 1}) {r.spec} & " + " & ".join(cells) + r" \\")
    if i == 0:
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("T_robust.tex", "\n".join(L))

# ---------------------------------------------------------------------------- A2. typology
ty = R["typology"]
L = [r"\begin{tabular}{@{} l cccc @{}}", r"\toprule",
     r"\textbf{Typology} & \textbf{Chile} & \textbf{Mexico} & \textbf{Colombia} & \textbf{Mean group} \\", r"\midrule"]
for tname in TYPOLOGIES:
    sub = ty[ty.typology == tname]
    cells = []
    for nm in COUNTRIES + ["Mean group"]:
        r = sub[sub.country == nm].iloc[0]
        cells.append(f"{f(r.est)}{star(r.p)} ({r.se:.2f})")
    L.append(f"{tname} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("T_typology.tex", "\n".join(L))
TY_MINHOLM = float(ty.p_holm.min())
TY_MINP = float(ty[ty.country != "Mean group"].p.min())

# ---------------------------------------------------------------------------- A3. legacy levels
lg = R["legacy"]
L = [r"\begin{tabular}{@{} L{7.6cm} ccccc @{}}", r"\toprule",
     r"\textbf{Specification} & \textbf{Coef.} & \textbf{S.e.} & $p$ & \textbf{Obs.} & \textbf{DW} \\", r"\midrule"]
for i, r in lg.iterrows():
    L.append(f"{r.spec} & {f(r.b, 4)} & ({r.se:.4f}) & {pv(r.p)} & {int(r.n)} & {r.dw:.2f} \\\\")
    if i in (4,):
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("T_legacy.tex", "\n".join(L))

# ---------------------------------------------------------------------------- A4. Granger robustness
gr = R["granger_rob"]
names = ["Baseline", "Excluding Mar--Dec 2020", "Outlier months trimmed", "Original pooled-normalized IFI"]
L = [r"\begin{tabular}{@{} l c cccc cccc @{}}", r"\toprule",
     r" & & \multicolumn{4}{c}{\textbf{FRISK $\to$ IFI}} & \multicolumn{4}{c}{\textbf{IFI $\to$ FRISK}} \\",
     r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
     r"\textbf{Sample / variable} & \textbf{Lags} & Chile & Mexico & Colombia & Joint & Chile & Mexico & Colombia & Joint \\", r"\midrule"]
for nm in names:
    for p_ in (3, 6):
        a = gr[(nm, "FRISK->IFI", p_)]
        b = gr[(nm, "IFI->FRISK", p_)]
        va = [a[c]["p"] for c in COUNTRIES] + [a["Joint (3 countries)"]["p"]]
        vb = [b[c]["p"] for c in COUNTRIES] + [b["Joint (3 countries)"]["p"]]
        L.append(f"{nm if p_ == 3 else ''} & {p_} & " + " & ".join(f"{v:.3f}" for v in va + vb) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("T_granger_rob.tex", "{\\setlength{\\tabcolsep}{3.5pt}\n" + "\n".join(L) + "}")

# ---------------------------------------------------------------------------- A5. Monte Carlo
mc2 = pd.read_csv(OUT / "mc_size2.csv") if (OUT / "mc_size2.csv").exists() else None
L = [r"\begin{tabular}{@{} L{7.6cm} cc @{}}", r"\toprule",
     r"\textbf{Test} & \textbf{(A) Trend-stat.} & \textbf{(B) Stoch. trend} \\", r"\midrule",
     r"\multicolumn{3}{@{}l}{\textit{Levels regressions (IFI on FRISK, two-way fixed effects)}} \\"]
a, b = mc.iloc[0], mc.iloc[1]
def row(lab, k, src=None):
    x, y = (a, b) if src is None else src
    L.append(f"{lab} & {x[k]:.3f} & {y[k]:.3f} \\\\")
row("Clustered by country, $t(2)$ critical values (old procedure)", "old_t2")
row("Clustered by country, normal critical values", "old_norm")
row("Driscoll--Kraay, bandwidth 12", "old2_dk")
L.append(r"\midrule")
L.append(r"\multicolumn{3}{@{}l}{\textit{First differences, local projections (new procedure): one restriction}} \\")
row("Average response $h=1,\dots,6$: Chile", "new_chile")
row("Average response $h=1,\dots,6$: Mexico", "new_mexico")
row("Average response $h=1,\dots,6$: Colombia", "new_colombia")
row("Average response $h=1,\dots,6$: mean group", "new_mg")
if mc2 is not None:
    a2, b2 = mc2.iloc[0], mc2.iloc[1]
    row("Cumulative effect, distributed lags: by country (average)", "fdcum_c", (a2, b2))
    row("Cumulative effect, distributed lags: pooled (Driscoll--Kraay)", "fdcum_pool", (a2, b2))
L.append(r"\midrule")
L.append(r"\multicolumn{3}{@{}l}{\textit{Tests of several restrictions with HAC covariance (not used)}} \\")
row(r"Six responses $\beta_{c,h}$ jointly zero (per country, average)", "lp_joint")
row("Seven distributed-lag coefficients jointly zero (per country, average)", "fd_joint_c")
row("Seven distributed-lag coefficients jointly zero (pooled)", "fd_joint_pool")
row("Granger, three lags (per country, average)", "gr3_c")
row("Granger, three lags (joint, 3 countries)", "gr3_joint")
row("Granger, six lags (per country, average)", "gr6_c")
row("Granger, six lags (joint, 3 countries)", "gr6_joint")
if mc2 is not None:
    L.append(r"\midrule")
    L.append(r"\multicolumn{3}{@{}l}{\textit{Other procedures used in the application}} \\")
    row("Sup-$t$ bands, any horizon: Chile", "sup_chile", (a2, b2))
    row("Sup-$t$ bands, any horizon: Mexico", "sup_mexico", (a2, b2))
    row("Sup-$t$ bands, any horizon: Colombia", "sup_colombia", (a2, b2))
    row("Equal average responses across countries (two restrictions)", "hom", (a2, b2))
    row("Wild-bootstrap Granger, three lags (per country, average)", "gw3_c", (a2, b2))
    row("Wild-bootstrap Granger, three lags (joint)", "gw3_j", (a2, b2))
    row("Wild-bootstrap Granger, six lags (per country, average)", "gw6_c", (a2, b2))
    row("Wild-bootstrap Granger, six lags (joint)", "gw6_j", (a2, b2))
L += [r"\bottomrule", r"\end{tabular}"]
write("T_mc.tex", "\n".join(L))
print("tables written; typology min raw p = %.3f, min Holm p = %.3f; FD hom p = %.3f; FD MG = %s" % (TY_MINP, TY_MINHOLM, FD_HOM_P, FD_MG))
