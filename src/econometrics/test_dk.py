"""Cross-check: pooled Driscoll-Kraay of frisk_lib.hac_joint vs linearmodels (kernel covariance)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
from frisk_lib import *
panel = load_panel()
rows = []
for c in COUNTRIES:
    g = panel[c]
    d = pd.DataFrame({'dy': g.ifi.diff(), 'dx': g.r.diff()}).dropna()
    d['country'] = c; rows.append(d)
d = pd.concat(rows)
# country-demeaned regression, my DK
for L in (4, 8):
    dm = d.copy()
    for v in ('dy', 'dx'):
        dm[v] = dm[v] - dm.groupby('country')[v].transform('mean')
    X = dm[['dx']].values; y = dm['dy'].values
    b, u = ols(X, y)
    idx = np.array([GRID.get_loc(t) for t in dm.index])
    V, _ = hac_joint([dict(X=X, u=u, idx=idx)], L, small_sample=False)
    dd = d.reset_index().rename(columns={'index': 'date'}).set_index(['country', 'date'])
    r = PanelOLS(dd['dy'], dd[['dx']], entity_effects=True).fit(cov_type='kernel', kernel='bartlett', bandwidth=L)
    print(f'L={L}: mine b={b[0]:.6f} se={np.sqrt(V[0,0]):.6f} | linearmodels b={r.params["dx"]:.6f} se={r.std_errors["dx"]:.6f}')
