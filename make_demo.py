# make_demo.py - create a tiny demo catalog and FAISS index
import json
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from src.rag.embeddings import prepare_text
import os


OUT_PQ = "data_processed/demo.parquet"

# ✅ Make sure folder exists
os.makedirs(os.path.dirname(OUT_PQ), exist_ok=True)

# Example demo dataframe
demo = {
    "anime": ["Naruto", "One Piece", "Death Note"],
    "mood": ["adventurous", "exciting", "dark"]
}

pd.DataFrame(demo).to_parquet(OUT_PQ, index=False)
print(f"Saved demo dataset at {OUT_PQ}")

demo = [
  {"id":1,"title_english":"Clannad","description":"A touching high-school drama about family and friendship.","genres":["Drama","Romance"],"tags":["tears","slice-of-life"],"averageScore":85,"popularity":120000,"start_year":2007,"cover_large":""},
  {"id":2,"title_english":"Haikyuu!!","description":"Volleyball high-energy sports anime about teamwork and growth.","genres":["Sports","Comedy"],"tags":["sports","inspiring"],"averageScore":90,"popularity":300000,"start_year":2014,"cover_large":""},
  {"id":3,"title_english":"Spirited Away","description":"A fantasy film by Miyazaki about a girl's adventure in a spirit world.","genres":["Adventure","Fantasy"],"tags":["movie","classic"],"averageScore":95,"popularity":500000,"start_year":2001,"cover_large":""},
  {"id":4,"title_english":"Toradora!","description":"A romantic comedy-drama with strong emotional beats.","genres":["Romance","Comedy"],"tags":["romcom","school"],"averageScore":88,"popularity":150000,"start_year":2008,"cover_large":""},
  {"id":5,"title_english":"Mushishi","description":"A calm, episodic, atmospheric series for chill moods.","genres":["Slice of Life","Mystery"],"tags":["iyashikei","calm"],"averageScore":89,"popularity":80000,"start_year":2005,"cover_large":""}
]

OUT_META = Path("data_processed/metadata.json")
OUT_PQ = Path("data_processed/anime_catalog.parquet")
OUT_FAISS = Path("faiss_index/index.faiss")

# save parquet
pd.DataFrame(demo).to_parquet(OUT_PQ, index=False)
# embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [prepare_text(r) for r in demo]
emb = model.encode(texts, convert_to_numpy=True)
np.save("data_processed/embeddings.npy", emb)
faiss.normalize_L2(emb)
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)
OUT_FAISS.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(OUT_FAISS))
with open(OUT_META, "w", encoding="utf-8") as f:
    json.dump(demo, f, ensure_ascii=False, indent=2)
print("Demo created: data_processed/anime_catalog.parquet and faiss_index/index.faiss")
