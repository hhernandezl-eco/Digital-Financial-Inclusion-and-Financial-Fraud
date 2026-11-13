This folder is empty in the repository on purpose. It is where the scripts in `src/collection` and `src/nlp` read and write
the corpus of posts, the embeddings, and the classifier. Posts are written by individual users, so the text, handles, and
profile information are not distributed. Everything downstream of `src/nlp/03_aggregate_counts.py` uses monthly counts, which
are in `data/public/frisk/`.
