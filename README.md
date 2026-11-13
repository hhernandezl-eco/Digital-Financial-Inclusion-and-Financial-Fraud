# Fraud visibility on social media and financial inclusion in Latin America

Replication materials for *Measuring Online Fraud Visibility via Social Media: The FRISK Index and an Application to Financial Inclusion in Latin America*.

The project builds FRISK, a monthly index of self-reported fraud complaints posted on X in Brazil, Mexico, Colombia, and Chile (2014-2024), and uses it in two ways: a comparison with the complaint records of Colombia's financial supervisor, and a time-series application on a financial inclusion index for Chile, Mexico, and Colombia.

## What is shared and what is not

The posts were written by individual users, so the text, handles, and profile information are **not** distributed (`data/private` is empty and ignored by git). What is shared is everything needed to reproduce the results from the point where posts become monthly counts:

| Stage | Needs private data | In this repository |
|---|---|---|
| 1. Collection of posts | no (needs API access) | `src/collection` |
| 2. Classification, embeddings, clustering | yes | `src/nlp` |
| 3. Counts by country, month, and typology | yes | `src/nlp/03_aggregate_counts.py` (output in `data/public/frisk`) |
| 4. FRISK index, financial inclusion index, econometrics, figures and tables | **no** | `src/econometrics` |

Stage 4 is the part that most readers will want to run, and it only uses the public files in `data/public`.

## Layout

```
data/
  public/            counts, banking aggregates, macro series, keyword dictionary (all redistributable)
  private/           local only: posts, embeddings, classifier (not tracked)
  derived/           files written by the scripts (not tracked)
src/
  collection/        query builder and collection script for the X API
  nlp/               classifier fine-tuning, embeddings, clustering, aggregation to counts
  econometrics/      index construction, estimation, Monte Carlo study, figures and tables
results/             csv files, tables, and figures produced by the code
docs/                data dictionary
```

## Reproducing the results (no private data needed)

```bash
pip install -r requirements.txt
cd src/econometrics

python build_frisk_index.py     # FRISK from monthly counts; checks against the stored index
python build_ifi.py             # Index of Financial Inclusion from the raw bank aggregates; lists breaks in results/ifi_breaks.csv
python sfc_validation.py        # FRISK against the supervisor's complaints (levels, detrended, differences)
python run_all.py               # unit roots, distributed lags, local projections, Granger tests, robustness, typologies (a few minutes)
python make_tables.py && python make_figures.py && python fig_ifi.py && python make_fig46.py
```

The Monte Carlo study of Appendix E takes longer (about an hour):

```bash
python mc_size.py 2000
python mc_size2.py 1000 300 199 A & python mc_size2.py 1000 300 199 B
python merge_mc2.py
```

The results of both studies are already saved in `results/`. `test_lib.py` and `test_dk.py` check the covariance estimator against `statsmodels` and `linearmodels`.

## Rebuilding the corpus

This needs API access with full-archive search and enough local storage for the embeddings.

```bash
export X_BEARER_TOKEN=...        # never commit credentials
export PSEUDONYM_SALT=...        # private string used to hash author ids at collection time

cd src/collection
python collect_posts.py --lang es --start 2014-01-01 --end 2024-12-31 --out ../../data/private/raw/es
python collect_posts.py --lang pt --start 2014-01-01 --end 2024-12-31 --out ../../data/private/raw/pt

cd ../nlp
python 00_finetune_classifier.py   # needs data/private/annotated_posts.csv (text, label)
python 01_embeddings.py
python 01b_predict.py
python 02_clustering.py           # DBSCAN on the 768-d embeddings of the flagged posts: cosine distance, eps 0.5, min_samples 5
# assign country (profile location) and typology (manual mapping of clusters) -> data/private/classified.csv
python 03_aggregate_counts.py
```

`build_queries.py` turns the keyword dictionary into search queries (18 per language), so the queries can be inspected without any access:

```bash
python src/collection/build_queries.py data/public/keywords/keywords.xlsx es
```

The country of each post is assigned from the free-text location in the author's profile; the assignment rules are described in the paper (Section 3.2) and are not reproduced by a script here.

## Main results

All numbers come from the files in `results/`.

| Quantity | Value |
|---|---|
| Correlation of FRISK with the supervisor's complaints, levels (Jan-2014 to Dec-2021, N = 96) | 0.29 |
| Same, after linear detrending | -0.15 |
| Same, monthly first differences | -0.01 |
| Mean-group response of the inclusion index, average over 1-6 months (s.d. of its monthly change) | +0.07 (s.e. 0.13) |
| Chile / Mexico / Colombia | -0.10 / +0.13 / +0.19 |

## Notes on the data

- The inclusion index is rebuilt from the raw files by `build_ifi.py`. Each component is rescaled within country because the files report accounts and ATMs per person in different units. The supervisor's account and deposit series for Colombia change coverage in July 2022 and November 2023; the script detects those breaks and splices them, and writes every adjustment to `results/ifi_breaks.csv`.
- Consumer prices for Colombia come from DANE (`ipc_colombia_monthly.csv`). To rebuild that file, place the monthly DANE annexes in `data/public/macro/dane_ipc_raw/` and run `build_ipc_colombia.py`.
- Numbers in the paper can be traced to `results/` and the scripts that write them.

## Software

Python 3.9 or later. Package versions are listed in `requirements.txt`.

## Contact

Hernando Hernández-Lasso (hhernandez@unal.edu.co).
