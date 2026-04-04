# src/rag/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
import json

CATALOG = Path("data_processed/anime_catalog.parquet")
FAISS_DIR = Path("faiss_index")
EMB_PATH = FAISS_DIR / "embeddings.npy"
INDEX_PATH = FAISS_DIR / "anime.index"
ID_MAP_PATH = FAISS_DIR / "id_map.json"
META_PATH = Path("data_processed/metadata.json")


def build_embeddings(model_name="all-MiniLM-L6-v2"):
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    df = pd.read_parquet(CATALOG)
    records = df.to_dict(orient="records")

    texts = []
    for r in records:
        title = r.get("title") or ""
        genres = r.get("genres") or ""
        desc = (r.get("description") or "")[:300]
        texts.append(f"{title}. Genres: {genres}. {desc}")

    print(f"Encoding {len(texts)} anime…")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMB_PATH, embeddings)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    id_map = {str(i): r.get("id") for i, r in enumerate(records)}
    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Built FAISS index with {index.ntotal} entries -> {INDEX_PATH}")


if __name__ == "__main__":
    build_embeddings()
