"""
Collects public posts that match the keyword dictionary, one day at a time, with the full-archive search endpoint of the
X API v2 (Tweepy). Access to that endpoint requires an API tier that includes full-archive search; the bearer token is read
from the environment and is never stored in the repository.

    export X_BEARER_TOKEN=...            (PowerShell: $env:X_BEARER_TOKEN = "...")
    export PSEUDONYM_SALT=...            (any private string; used to pseudonymize author ids)
    python collect_posts.py --lang es --start 2014-01-01 --end 2014-01-31 --out ../../data/private/raw/es

Output: one JSON-lines file per month in --out with the fields
    post_id, created_at, lang, text, author_hash, profile_location
Author identifiers are replaced by a salted SHA-256 hash at collection time; handles and user names are not stored.
The files in data/private are excluded from version control (see .gitignore).
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tweepy

from build_queries import build_queries

FIELDS = ["created_at", "lang", "author_id"]
USER_FIELDS = ["location"]


def pseudonym(author_id: str, salt: str) -> str:
    return hashlib.sha256((salt + str(author_id)).encode("utf-8")).hexdigest()[:16]


def collect_day(client, query: str, day: datetime, salt: str):
    start, end = day, day + timedelta(days=1)
    pages = tweepy.Paginator(
        client.search_all_tweets, query=query, start_time=start, end_time=end,
        tweet_fields=FIELDS, user_fields=USER_FIELDS, expansions=["author_id"], max_results=500)
    for page in pages:
        if page.data is None:
            continue
        users = {u.id: u for u in (page.includes or {}).get("users", [])}
        for t in page.data:
            loc = users[t.author_id].location if t.author_id in users else None
            yield {"post_id": str(t.id), "created_at": t.created_at.isoformat(), "lang": t.lang, "text": t.text,
                   "author_hash": pseudonym(t.author_id, salt), "profile_location": loc}
        time.sleep(1.1)          # stay inside the rate limit of the full-archive endpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["es", "pt"], required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keywords", default=str(Path(__file__).resolve().parents[2] / "data" / "public" / "keywords" / "keywords.xlsx"))
    a = ap.parse_args()

    client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"], wait_on_rate_limit=True)
    salt = os.environ["PSEUDONYM_SALT"]
    queries = build_queries(a.keywords, a.lang)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    day, last = [datetime.fromisoformat(x).replace(tzinfo=timezone.utc) for x in (a.start, a.end)]
    seen = set()
    while day <= last:
        path = out / f"{day:%Y-%m}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for q in queries:
                for rec in collect_day(client, q, day, salt):
                    if rec["post_id"] not in seen:          # a post can match several query blocks
                        seen.add(rec["post_id"]); f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        day += timedelta(days=1)


if __name__ == "__main__":
    main()
