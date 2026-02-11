# src/rag/retriever.py
import faiss, numpy as np, json
from sentence_transformers import SentenceTransformer
from pathlib import Path

FAISS_PATH = Path("faiss_index/index.faiss")
META_PATH = Path("data_processed/metadata.json")
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index():
    index = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metas = json.load(f)
    return index, metas

def retrieve(query, top_k=10):
    vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    index, metas = load_index()
    D, I = index.search(vec, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        meta = metas[idx]
        results.append({"score": float(dist), "meta": meta})
    return results
