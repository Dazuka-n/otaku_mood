# src/rag/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
import json
import re

CATALOG = Path("data_processed/anime_catalog.parquet")
EMB_PATH = Path("data_processed/embeddings.npy")
META_PATH = Path("data_processed/metadata.json")
FAISS_PATH = Path("faiss_index/index.faiss")

def clean_text(s):
    if not s: return ""
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("\n"," ").strip()
    return s

def prepare_text(row):
    title = row.get("title_english") or row.get("title_romaji") or ""
    synopsis = clean_text(row.get("description") or "")
    tags = " ".join(row.get("tags") or []) if row.get("tags") else ""
    genres = " ".join(row.get("genres") or []) if row.get("genres") else ""
    text = f"{title}. {synopsis[:600]}. Genres: {genres}. Tags: {tags}"
    return text

def build_embeddings(model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    df = pd.read_parquet(CATALOG)
    texts = [prepare_text(r) for r in df.to_dict(orient="records")]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMB_PATH, embeddings)
    metas = df.to_dict(orient="records")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metas, f)
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    print("Built index with", index.ntotal)

if __name__ == "__main__":
    build_embeddings()
