# OtakuMood

> *Not just anime. A mood.*

OtakuMood is a mood-driven anime recommendation engine that understands how you're feeling and surfaces the right anime for that exact moment. Instead of filtering by genre or year, you describe your vibe in plain language — a sentence, an emoji, a feeling — and the system figures out the rest.

---

## How It Works

User input flows through a three-stage pipeline:

1. **Mood classification** — A Groq-hosted LLM (Llama 3.3 70B) reads your text and classifies it into one of eight emotional categories: `happy`, `sad`, `chill`, `excited`, `anxious`, `romantic`, `dark`, or `nostalgic`.

2. **Semantic search** — Your query is embedded using `all-MiniLM-L6-v2` and run against a FAISS vector index built from 500+ anime titles sourced via the AniList GraphQL API. Each anime is embedded from a composite of its title, genres, and synopsis.

3. **Mood-aware reranking** — The top 25 FAISS candidates are reranked with a weighted formula: 60% semantic similarity + 30% genre-to-mood alignment boost + 10% popularity. Results are then filtered by your year range and minimum score settings.

Each result renders as a card with poster art, score badge, genre tags, synopsis, and a "Why this matches you" expander. You can click **Explain** to get an LLM-generated explanation of the match, or **Save** to add it to your Favorites Shrine.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + custom CSS (dark glassmorphic theme) |
| LLM / Mood detection | Groq API — `llama-3.3-70b-versatile` |
| LLM explanations | LangChain + Groq |
| Semantic search | FAISS (`IndexFlatIP`) + `all-MiniLM-L6-v2` |
| Anime data | AniList GraphQL API (500+ titles, cover art, metadata) |
| Storage | Parquet catalog, `.npy` embeddings, `favorites.json` |
| Runtime | Python 3.11, PyTorch, HuggingFace Transformers |

---

## Project Structure

```
otaku_mood/
├── src/
│   ├── app/
│   │   └── app.py              # Streamlit UI (~700 lines)
│   ├── rag/                    # Retriever, reranker, LLM chain
│   ├── training/               # BERT fine-tuning + synthetic data gen
│   ├── data/
│   │   └── fetch_api.py        # AniList GraphQL fetcher (500+ anime)
│   └── preprocessing/          # Raw data → Parquet → embeddings
├── data_processed/
│   ├── anime_catalog.parquet   # Anime metadata + cover URLs
│   └── favorites.json          # Saved anime (user shrine)
├── faiss_index/
│   ├── anime.index             # FAISS vector index
│   ├── embeddings.npy          # Sentence-transformer embeddings
│   └── id_map.json             # FAISS position → anime ID mapping
├── models/                     # Fine-tuned BERT checkpoint (optional)
├── make_demo.py                # Builds a minimal 5-anime demo index
├── requirements.txt
└── .streamlit/
    └── secrets.toml            # API keys (not committed)
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier works)

### 1. Clone and install

```bash
git clone https://github.com/your-username/otaku-mood.git
cd otaku-mood

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Add your API key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 3a. Run with demo data (fastest)

Builds a minimal 5-anime index — no API calls required:

```bash
python make_demo.py
python -m streamlit run src/app/app.py
```

### 3b. Run with full data (recommended)

Fetches 500+ anime from AniList and builds the real FAISS index:

```bash
# Fetch anime catalog from AniList GraphQL API
python src/data/fetch_api.py

# Build embeddings and FAISS index
python src/preprocessing/build_index.py

# Launch app
python -m streamlit run src/app/app.py
```

---

## Features

**Mood input** — Free text, emojis, or preset vibes (Cozy, Hype, Mind-bend, Tender). The LLM interprets the emotional intent regardless of phrasing.

**Era and score filters** — Sidebar sliders for year range (1980–2025) with quick presets (90s Classics, 2000s Era, Modern Hits) and minimum score threshold (default: 70).

**Override mood** — If the auto-detected mood isn't right, pick one manually from the dropdown.

**LLM explanations** — Click Explain on any card to get a Groq-generated paragraph explaining why that specific anime matches your current mood.

**Favorites shrine** — Save any anime; persists across sessions in `data_processed/favorites.json`.

---

## Configuration

All configuration is managed via `.streamlit/secrets.toml` and environment variables:

| Key | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for mood detection and explanations | Yes |
| `DEMO_VIDEO_PATH` | Path to a demo video file to surface in the UI | No |

---

## Design Decisions

**Why Groq instead of BERT for mood classification?**
The original design used a fine-tuned `bert-base-uncased` classifier with 8 mood labels. In deployment, the model checkpoint failed to load — the training artefact wasn't portable across environments. Rather than debugging the ML training pipeline, the mood classification was migrated to a zero-shot Groq LLM call with a structured prompt. This is faster to iterate on, requires no checkpoint management, and handles edge cases and mixed-emotion inputs that a fixed classifier struggles with.

**Why FAISS + reranking instead of pure semantic search?**
Pure cosine similarity returns "similar" content, not "mood-appropriate" content. A query like "something cozy and nostalgic" might semantically match any slice-of-life anime regardless of tone. The reranker adds a genre boost layer that weights results toward genres statistically associated with each mood — for example, `dark` queries boost Psychological, Thriller, and Horror genres. The 60/30/10 weighting was chosen to keep semantic relevance dominant while allowing mood alignment to meaningfully shift rankings.

---

## Requirements

See `requirements.txt` for the full list. Key dependencies:

```
streamlit
langchain
langchain-groq
groq
faiss-cpu
sentence-transformers
torch
transformers
pandas
pyarrow
numpy
```

---

## License

MIT
