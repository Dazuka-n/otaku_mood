# src/preprocessing/preprocess_catalog.py
import pandas as pd
import json
from pathlib import Path

RAW_DIR = Path("data_raw")
OUT = Path("data_processed")
OUT.mkdir(exist_ok=True, parents=True)

def unify_and_save(anilist_json_list, out_path=OUT/"anime_catalog.parquet"):
    rows=[]
    for m in anilist_json_list:
        rows.append({
            "id": m.get("id"),
            "idMal": m.get("idMal"),
            "title_romaji": (m.get("title") or {}).get("romaji"),
            "title_english": (m.get("title") or {}).get("english"),
            "description": m.get("description"),
            "genres": m.get("genres"),
            "tags": [t.get("name") for t in m.get("tags",[])] if m.get("tags") else [],
            "averageScore": m.get("averageScore"),
            "popularity": m.get("popularity"),
            "start_year": (m.get("startDate") or {}).get("year"),
            "siteUrl": m.get("siteUrl"),
            "cover_large": (m.get("coverImage") or {}).get("large"),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print("Saved", out_path)
