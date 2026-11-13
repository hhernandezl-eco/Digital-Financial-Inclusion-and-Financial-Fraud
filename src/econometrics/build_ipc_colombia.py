"""Monthly CPI of Colombia (DANE) 2013-2022 from the files in the DANE folder:
 - IPC_Variacion.xls (converted to xlsx): monthly % changes 2003-2018 (base Dec-2008=100)
 - anexo_ipc_<mes><aa>.xlsx, sheet '2', row 'Total', column 'Mensual': monthly % changes 2019-2022.
Output data/ipc_colombia_monthly.csv with the monthly change and a chained index (Dec-2013 = 100)."""
import glob, re
import numpy as np, pandas as pd, openpyxl
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
MES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
ws = openpyxl.load_workbook(ROOT / "data" / "public" / "macro" / "dane_ipc_raw" / "IPC_Variacion_converted.xlsx", data_only=True).worksheets[0]
rows = list(ws.iter_rows(values_only=True))
hdr = next(i for i, r in enumerate(rows) if r[0] == "Mes")
years = rows[hdr][1:]
rec = []
for k in range(12):
    r = rows[hdr + 1 + k]
    for y, v in zip(years, r[1:]):
        if y and 2013 <= int(y) <= 2018 and v is not None:
            rec.append((int(y), k + 1, float(v)))
for f in glob.glob(str(ROOT / "data" / "public" / "macro" / "dane_ipc_raw" / "anex*.xlsx")):
    m = re.search(r"anex(?:o_ipc_|-IPC-)([a-z]{3})(\d{2,4})", f)
    mo, yy = MES.index(m.group(1)) + 1, (2000 + int(m.group(2)) if len(m.group(2)) == 2 else int(m.group(2)))
    w = openpyxl.load_workbook(f, data_only=True)["2"]
    tot = [r for r in w.iter_rows(values_only=True) if r[0] and str(r[0]).strip().startswith("Total")][0]
    rec.append((yy, mo, float(tot[2])))
d = pd.DataFrame(rec, columns=["year", "month", "dcpi_pct"]).sort_values(["year", "month"]).reset_index(drop=True)
assert len(d) == 144 and not d.duplicated(["year", "month"]).any()
d["cpi_idx"] = 100 * np.cumprod(1 + d.dcpi_pct / 100)
d.to_csv(ROOT / "data" / "public" / "macro" / "ipc_colombia_monthly.csv", index=False)
ann = {y: (np.prod(1 + g.dcpi_pct / 100) - 1) * 100 for y, g in d.groupby("year")}
dane = {2013: 1.94, 2014: 3.66, 2015: 6.77, 2016: 5.75, 2017: 4.09, 2018: 3.18, 2019: 3.8, 2020: 1.61, 2021: 5.62, 2022: 13.12, 2023: 9.28, 2024: 5.2}
for y in dane: print(y, round(ann[y], 2), dane[y])
