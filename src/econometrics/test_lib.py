"""Unit tests of the HAC machinery against statsmodels."""
import numpy as np, pandas as pd, statsmodels.api as sm
from frisk_lib import *
g = load_panel()['Chile']
des = make_design(g, p=3, H=6)
X, Y, idx = des['X'], des['Y'], des['idx']
ib = des['cols'].index('shock')
for h in (0, 3, 6):
    for L in (0, 4, 8):
        b, u = ols(X, Y[:, h])
        V, _ = hac_joint([dict(X=X, u=u, idx=idx)], L)
        m = sm.OLS(Y[:, h], X).fit(cov_type='HAC', cov_kwds=dict(maxlags=L, use_correction=True, kernel='bartlett')) if L>0 else sm.OLS(Y[:, h], X).fit(cov_type='HC1')
        print(h, L, 'mine se=%.6f  statsmodels se=%.6f' % (np.sqrt(V[ib, ib]), m.bse[ib]))
# consecutive sample check
print('rows', des['n'], 'consecutive:', (np.diff(idx) == 1).all())
# DK test: pooled = sum scores by date; compare with linearmodels-free manual
