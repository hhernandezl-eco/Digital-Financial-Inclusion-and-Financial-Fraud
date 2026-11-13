"""mc_size2.py -- additional size checks: sup-t bands, homogeneity test, FD cumulative effect, wild-bootstrap Granger."""
import sys
import numpy as np, pandas as pd
from mc_size_dgp import *
from frisk_est import *
NREP = int(sys.argv[1]); NG = int(sys.argv[2]); B = int(sys.argv[3])
KINDS = tuple(sys.argv[4]) if len(sys.argv) > 4 else ("A", "B")
rows = []
for kind in KINDS:
    r = dict(sup=[0, 0, 0], hom=0, fdcum_c=0, fdcum_pool=0, mg_fd=0)
    g3 = dict(c=0, j=0); g6 = dict(c=0, j=0); ng = 0
    for rep in range(NREP):
        ifi, fr = simulate(kind)
        pn = to_panel(ifi, fr)
        res = lp_system(pn); sm = lp_summary(res)
        for ci, c in enumerate(COUNTRIES):
            Vc, crit = sup_bands(res, ci)
            t = res["beta"][ci] / np.sqrt(np.diag(Vc))
            r["sup"][ci] += np.max(np.abs(t)) > crit
        r["hom"] += sm[("hom", "avg")]["p_F"] < 0.05
        fd, _, _ = fd_dl(pn)
        for c in COUNTRIES + ["Pooled"]:
            th, V = fd[c]["theta"], fd[c]["V"]
            tt = th.sum() / np.sqrt(np.ones(7) @ V @ np.ones(7))
            if c == "Pooled":
                r["fdcum_pool"] += abs(tt) > 1.96
            else:
                r["fdcum_c"] += (abs(tt) > 1.96) / 3
        if rep < NG:
            ng += 1
            for p_, store in ((3, g3), (6, g6)):
                gg = granger_wild(pn, "r", "ifi", p_, B=B, seed=1000 + rep)
                store["c"] += np.mean([gg[c]["p"] < 0.05 for c in COUNTRIES])
                store["j"] += gg["Joint (3 countries)"]["p"] < 0.05
    n = NREP
    rows.append(dict(dgp=kind, sup_chile=r["sup"][0] / n, sup_mexico=r["sup"][1] / n, sup_colombia=r["sup"][2] / n, hom=r["hom"] / n,
                     fdcum_c=r["fdcum_c"] / n, fdcum_pool=r["fdcum_pool"] / n,
                     gw3_c=g3["c"] / ng, gw3_j=g3["j"] / ng, gw6_c=g6["c"] / ng, gw6_j=g6["j"] / ng, nrep=n, nrep_g=ng, B=B))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / (f"mc_size2_{''.join(KINDS)}.csv"), index=False)
