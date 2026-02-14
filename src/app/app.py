# src/app/app.py
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from src.utils.config import load_env
from src.rag.retriever import retrieve
from src.rag.scoring import rerank
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
ENV = load_env()

st.set_page_config(page_title="OtakuMood", layout="wide")

# ------------------- Added: Load custom CSS -------------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("src/app/style.css")
# ---------------------------------------------------------------

# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@st.cache_resource
def load_models():
    mood_model_dir = ENV.get("MOOD_CLASSIFIER_PATH", "models/mood_intent_classifier/mood")
    tokenizer = AutoTokenizer.from_pretrained(mood_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(mood_model_dir)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=-1)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return pipe, embed_model

# Attempt to load, with graceful fallback
try:
    mood_pipe, embed_model = load_models()
except Exception as e:
    mood_pipe = None
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

st.sidebar.title("OtakuMood")
manual_mood = st.sidebar.selectbox("Override mood (optional)", ["(auto)", "happy", "chill", "sad", "angry", "romantic", "adventurous", "nostalgic", "inspiring"])
year_min, year_max = st.sidebar.slider("Year range", 1980, 2025, (1990, 2025))
min_score = st.sidebar.slider("Min average score", 0, 100, 50)

# ------------------- Added: Hero Section -------------------
st.image("src/app/heroarea.png", caption="OtakuMood Logo")
st.markdown("""
<div class="hero">
    <img src="C:/Users/Krishna/Desktop/otaku_mood/src/app/heroarea.png" alt="OtakuMood Logo">
    <h1>OtakuMood</h1>
    <p>Your neon-lit anime recommender — mood-based, stylish, and fun!</p>
</div>
""", unsafe_allow_html=True)
# -------------------------------------------------------------

st.title("OtakuMood — Anime Recommender by Mood")
st.write("Type how you feel or what you want to watch. OtakuMood will guess your mood and recommend anime.")

user_input = st.text_input("Tell me how you're feeling or ask a question:", placeholder="I'm feeling nostalgic and want something chill...")

def run_chain_groq(query, mood):
    """LLM explanation using Groq."""
    prompt = f"""The user feels '{mood}' and wrote: '{query}'.
Explain briefly why the recommended anime matches their mood and preferences."""
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # you can switch to "mixtral-8x7b-32768" if available
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Groq error: {e})"

if user_input:
    if manual_mood != "(auto)":
        mood = manual_mood
        st.info(f"Manual mood override: {mood}")
    else:
        if mood_pipe:
            pred = mood_pipe(user_input, top_k=1)[0]
            mood = pred['label']
            st.success(f"Predicted mood: {mood}")
        else:
            mood = "chill"
            st.warning("Mood model not found, defaulting to 'chill'")

    candidates = retrieve(user_input, top_k=25)
    reranked = rerank(candidates, mood)[:10]

    filtered = []
    for c in reranked:
        meta = c['meta']
        yr = meta.get('start_year') or 0
        score = meta.get('averageScore') or 0
        if yr >= year_min and yr <= year_max and score >= min_score:
            filtered.append(c)

    # ------------------- Added: Recommendations as Cards -------------------
    st.subheader("Top recommendations")
    cols = st.columns(2)
    for i, item in enumerate(filtered[:10]):
        meta = item['meta']
        col = cols[i % 2]
        with col:
            cover_url = meta.get("cover_large")
            genres = ', '.join(meta.get('genres') or [])
            title = meta.get('title_english') or meta.get('title_romaji')
            score = meta.get('averageScore')
            desc = (meta.get("description") or "")[:350] + "..."

            st.markdown(f"""
            <div class="card">
                <img src="{cover_url}" width="100%" />
                <h3>{title}</h3>
                <p><b>Genres:</b> {genres}</p>
                <p><b>Score:</b> {score}</p>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns([1, 3])
            if c1.button(f"Explain why {i}", key=f"explain_{i}"):
                explanation = run_chain_groq(user_input, mood)
                st.write(explanation)
            if c2.button(f"Save favorite {i}", key=f"fav_{i}"):
                fav_file = Path("data_processed/favorites.json")
                favs = []
                if fav_file.exists():
                    favs = json.loads(fav_file.read_text(encoding="utf-8"))
                favs.append({"meta": meta})
                fav_file.write_text(json.dumps(favs, ensure_ascii=False, indent=2), encoding="utf-8")
                st.success("Saved to favorites")
    # ----------------------------------------------------------------------

    st.sidebar.subheader("Favorites")
    fav_file = Path("data_processed/favorites.json")
    if fav_file.exists():
        favs = json.loads(fav_file.read_text(encoding="utf-8"))
        for f in favs[-5:]:
            st.sidebar.write(f.get("meta", {}).get("title_english") or f.get("meta", {}).get("title_romaji"))
