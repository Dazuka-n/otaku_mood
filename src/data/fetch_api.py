# src/data/fetch_api.py
import requests
import time
import re
import pandas as pd
from pathlib import Path

ANILIST_API = "https://graphql.anilist.co"
OUT_PATH = Path("data_processed/anime_catalog.parquet")

QUERY = """
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    media(type: ANIME, sort: POPULARITY_DESC) {
      id
      title { english romaji }
      genres
      averageScore
      startDate { year }
      description(asHtml: false)
      coverImage { large }
      popularity
      episodes
      duration
    }
  }
}
"""


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("\n", " ").strip()


def fetch_page(page: int) -> list:
    r = requests.post(
        ANILIST_API,
        json={"query": QUERY, "variables": {"page": page}},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("data", {}).get("Page", {}).get("media", [])


def fetch_all(pages: int = 10, sleep_sec: float = 1.0) -> list:
    rows = []
    for page in range(1, pages + 1):
        print(f"Fetching page {page}/{pages}…")
        media = fetch_page(page)
        if not media:
            print(f"  No data on page {page}, stopping.")
            break
        for m in media:
            title_obj = m.get("title") or {}
            title = title_obj.get("english") or title_obj.get("romaji") or ""
            genres = ", ".join(m.get("genres") or [])
            rows.append({
                "id": m.get("id"),
                "title": title,
                "genres": genres,
                "score": m.get("averageScore"),
                "year": (m.get("startDate") or {}).get("year"),
                "description": clean_html(m.get("description") or ""),
                "cover_url": (m.get("coverImage") or {}).get("large"),
                "popularity": m.get("popularity"),
                "episodes": m.get("episodes"),
                "duration_mins": m.get("duration"),
            })
        time.sleep(sleep_sec)
    return rows


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = fetch_all(pages=10)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} anime to {OUT_PATH}")
