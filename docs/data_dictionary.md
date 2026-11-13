# Data dictionary

All files in `data/public` can be redistributed. None contains text, user names, handles, identifiers, or dates finer than a month.

## data/public/frisk
| File | Content |
|---|---|
| `monthly_counts.csv` | Posts classified as first-person fraud complaints, by country and month (Dec-2013 to Dec-2024), with population in thousands. Input of `build_frisk_index.py`. |
| `dataset_FRISK_panel_clean.csv` | Country-month panel for Chile, Mexico and Colombia: total posts, posts by typology (General Fraud Complaints, Social Engineering, Identity Theft, Payment Fraud, Account Takeover) and the standardized log index of each (`std_ln_FRISK_*`). |
| `FRISK_canonical_monthly.csv` | Monthly index for the four countries: `posts`, `rate` (per 100,000), `r` = log(1 + rate), `frisk_z` (standardized over the Chile-Mexico-Colombia panel). |
| `FRISK_canonical_annual.csv` | Annual averages of `frisk_z` and annual posts. |

## data/public/ifi
| File | Content |
|---|---|
| `components/IFI_CHILE.xlsx`, `IFI_MEX.xlsx`, `IFI_COL.xlsx` | Monthly banking aggregates by country: accounts, ATMs, deposits, credit, working-age population and GDP, taken from the national supervisory sources. `src/econometrics/build_ifi.py` rebuilds the Index of Financial Inclusion from these files. |
| `IFI_Final.csv`, `IFI_Final_pooled.csv` | Earlier versions of the index (within-country and pooled normalization). Kept only for the robustness checks and the benchmark regressions in levels. |

## data/public/macro
| File | Content |
|---|---|
| `Controles_Final.csv` | Monthly activity index (2015 = 100), unemployment rate, mobile connections, population and digital transactions by country. The price-index column of this file is not used; consumer prices come from the files below. |
| `ipc_colombia_monthly.csv` | Colombian consumer price index (DANE), monthly change and chained index (Dec-2013 = 100), 2013-2024, built by `build_ipc_colombia.py` from the DANE annexes. |
| `chile_cpi_fred.csv` | Chilean consumer price index (OECD series, 2015 = 100) to Dec-2023; 2024 is chained with the national index in `frisk_lib.py`. |

## data/public/sfc and indices
| File | Content |
|---|---|
| `sfc/colombia_official_vs_frisk.csv` | Monthly fraud complaints received by Colombia's financial supervisor (national total) next to FRISK, Dec-2013 to Dec-2021. The comparison in the paper uses January 2014 onward. |
| `indices/General_Corruption_Index.xlsx` | Annual Basel AML Index, Corruption Perceptions Index and OCIndex by country. |
| `keywords/keywords.xlsx` | Keyword dictionary used to build the search queries (sheet `Diccionario_Total`) with the interview terms and the category definitions. |
