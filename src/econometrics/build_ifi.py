"""
build_ifi.py -- traceable reconstruction of the Index of Financial Inclusion (IFI) from the raw component files.

Components (per country, monthly 2014-2024), all scale-free ratios so that unit differences between files cannot matter:
  penetration  = deposit accounts / working-age population            (Chile: checking accounts; Mexico: accounts; Colombia: accounts of natural persons)
  availability = ATMs / working-age population
  usage        = mean of  [deposits / GDP]  and  [credit / GDP]
Each component is rescaled to [0,1] WITHIN COUNTRY (min-max over the country's own sample) -- the raw files express
accounts and ATMs per capita in different units across countries, so a pooled min-max is not meaningful.
IFI = 1 - sqrt((1-d_pen)^2 + (1-d_avail)^2 + (1-d_use)^2) / sqrt(3)       (Sarma 2008)

Structural breaks in the Colombian supervisory series (changes in the reporting perimeter) are removed by SPLICING:
a month t is a break if |dlog x_t| > BREAK_THR and exceeds BREAK_MAD times the median absolute monthly change; the part of the
series before t is multiplied by the factor that makes the growth at t equal to the median growth of the 3 adjacent months.
Every break, factor and component is written to output/ifi_breaks.csv.
"""
import numpy as np, pandas as pd
from frisk_lib import ROOT, OUT, DATA
BREAK_THR, BREAK_MAD, OUT_THR = 0.12, 6.0, 0.25

def read(fn):
    d = pd.read_excel(ROOT / "data" / "public" / "ifi" / "components" / fn); d.columns = [c.replace("\ufffd", "?") for c in d.columns]
    a, b = d.columns[0], d.columns[1]
    if fn.startswith("IFI_MEX"): a, b = b, a
    d = d.rename(columns={a: "year", b: "month"}).dropna(subset=["year", "month"])
    d["year"], d["month"] = d.year.astype(int), d.month.astype(int)
    d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=1))
    return d.drop_duplicates("date").set_index("date").sort_index()

ch, mx, co = read("IFI_CHILE.xlsx"), read("IFI_MEX.xlsx"), read("IFI_COL.xlsx")
raw = {
 "Chile": pd.DataFrame({"acc": ch.Cuentas_corriente_chile, "atm": ch.Cajeros_chile, "pet": ch.PET_chile, "dep": ch.Valor_depositos_bancos_chile, "cred": ch.Cartera_creditos, "gdp": ch.PIB_chile_millones}),
 "Mexico": pd.DataFrame({"acc": mx.Numero_cuentas_MEX, "atm": mx.atm_MEX, "pet": mx.PET_MEX, "dep": mx.Depositos_MEX, "cred": mx.Cartera_creditos_MEX, "gdp": mx.PIB_MEX}),
 "Colombia": pd.DataFrame({"acc": co.Total_Cuentas_persona_natural, "atm": co.atm, "pet": co.PET, "dep": co.Total_Valor_persona_natural, "cred": co.saldo_cartera_total_fechacorte, "gdp": co.PIB_mensual}),
}
def splice(s, name, country, log):
    """(1) zeros -> missing; (2) one-month outliers (deviate >OUT_THR from BOTH neighbours, same sign) -> missing;
    (3) persistent breaks spliced backwards; (4) missing months interpolated linearly in logs."""
    s = s.astype(float).where(lambda v: v > 0)
    for t in s.index[~s.isna()]:
        loc = s.index.get_loc(t)
        if 0 < loc < len(s) - 1 and pd.notna(s.iloc[loc - 1]) and pd.notna(s.iloc[loc + 1]):
            a, b = np.log(s.iloc[loc] / s.iloc[loc - 1]), np.log(s.iloc[loc] / s.iloc[loc + 1])
            if min(abs(a), abs(b)) > OUT_THR and a * b > 0:
                log.append(dict(country=country, component=name, date=t.date(), kind="outlier -> missing", dlog=float(a), factor=np.nan)); s.iloc[loc] = np.nan
    for t in s.index[s.isna()]:
        if t not in [pd.Timestamp(x["date"]) for x in log if x["component"] == name and x["country"] == country]:
            log.append(dict(country=country, component=name, date=t.date(), kind="zero/missing", dlog=np.nan, factor=np.nan))
    v = s.dropna(); d = np.log(v).diff(); mad = d.abs().median()
    for t in d.index[1:]:
        if abs(d[t]) > BREAK_THR and abs(d[t]) > BREAK_MAD * mad:
            loc = v.index.get_loc(t); nb = pd.concat([d.iloc[max(1, loc - 3):loc], d.iloc[loc + 1:loc + 4]]).median()
            factor = float(np.exp(d[t] - nb)); v.iloc[:loc] = v.iloc[:loc] * factor
            d = np.log(v).diff()
            log.append(dict(country=country, component=name, date=t.date(), kind="break -> spliced", dlog=float(d.get(t, np.nan)), factor=factor))
    out = v.reindex(s.index)
    return np.exp(np.log(out).interpolate(limit_area="inside"))
log, out = [], {}
for c, r in raw.items():
    r = r[(r.index >= "2014-01-01") & (r.index <= "2024-12-01")].copy()
    if c == "Colombia":
        r = r[r.cred > 0]          # SFC loan portfolio is available from January 2015
        for k in ("acc", "dep", "cred", "atm"):
            r[k] = splice(r[k].astype(float), k, c, log)
    comp = pd.DataFrame({"pen": r.acc / r.pet, "avail": r.atm / r.pet, "dep": r.dep / r.gdp, "cred": r.cred / r.gdp})
    mm = (comp - comp.min()) / (comp.max() - comp.min())
    d_use = (mm.dep + mm.cred) / 2
    ifi = 1 - np.sqrt((1 - mm.pen) ** 2 + (1 - mm.avail) ** 2 + (1 - d_use) ** 2) / np.sqrt(3)
    out[c] = pd.DataFrame({"ifi_new": ifi, "d_pen": mm.pen, "d_avail": mm.avail, "d_use": d_use})
pd.DataFrame(log).to_csv(OUT / "ifi_breaks.csv", index=False)
res = pd.concat(out, names=["country", "date"]).reset_index(); res.to_csv(DATA / "ifi_rebuilt.csv", index=False)
print(pd.DataFrame(log).to_string()); print(res.groupby("country").ifi_new.agg(["count", "mean", "std", "min", "max"]))
