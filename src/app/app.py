# src/app/app.py
import base64
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from dotenv import load_dotenv
from groq import Groq

from src.utils.config import load_env
from src.rag.retriever import retrieve
from src.rag.scoring import rerank

load_dotenv()
ENV = load_env()

st.set_page_config(page_title="OtakuMood", layout="wide")

if "mood_input" not in st.session_state:
    st.session_state.mood_input = ""
if "year_range" not in st.session_state:
    st.session_state.year_range = (1990, 2025)
if "min_score" not in st.session_state:
    st.session_state.min_score = 50

SUGGESTIONS = [
    ("Cozy & nostalgic", "I'm feeling cozy and nostalgic. Recommend something warm and heartfelt."),
    ("Dark psychological", "I'm in the mood for something dark and psychological with twists."),
    ("Comfort romance", "I want a comfort romance anime with wholesome vibes."),
    ("High-energy action", "Give me a high-energy action anime to get hyped."),
    ("Slice-of-life zen", "I want a calm slice-of-life anime to unwind with tonight."),
]

PLACEHOLDERS = [
    "What kind of story do you want to step into tonight?",
    "Surprise me with ethereal fantasy vibes…",
    "Feeling burnt out — need healing anime recs",
    "Need a mind-bending thriller with twists",
]

VIBE_ADJUSTMENTS = [
    "More wholesome",
    "More intense",
    "Too dark",
]

DEMO_VIDEO_PATH = ENV.get("DEMO_VIDEO_PATH")
PROJECT_GUIDE_POINTS = [
    "Start by channeling your vibe in the main composer — adjectives, emojis, or full prompts all work.",
    "Fine-tune the era and score sliders on the left to match your nostalgia levels.",
    "Use Explain to open the full detail view and see why each anime matches your mood.",
    "Save favorites to build a mini altar of comfort rewatches.",
]

if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False
if "manual_mood_value" not in st.session_state:
    st.session_state.manual_mood_value = "(auto)"
if "year_range_value" not in st.session_state:
    st.session_state.year_range_value = (1990, 2025)
if "min_score_value" not in st.session_state:
    st.session_state.min_score_value = 50
if "show_filter_modal" not in st.session_state:
    st.session_state.show_filter_modal = False
if "show_mood_modal" not in st.session_state:
    st.session_state.show_mood_modal = False
if "show_favorites_modal" not in st.session_state:
    st.session_state.show_favorites_modal = False
if "trigger_results" not in st.session_state:
    st.session_state.trigger_results = False
if "show_detail_modal" not in st.session_state:
    st.session_state.show_detail_modal = False
if "detail_modal_payload" not in st.session_state:
    st.session_state.detail_modal_payload = None

def trigger_results_view(value: str | None = None, update_text: bool = False):
    if value is None:
        value = st.session_state.get("mood_input", "")
    if update_text:
        st.session_state.mood_input = value
    st.session_state.trigger_results = bool(value.strip())

# ------------------- Styling helpers -------------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def clean_snippet(text: str, limit: int = 420) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


local_css("src/app/style.css")
# -------------------------------------------------------

def open_detail_modal(payload: dict):
    st.session_state.detail_modal_payload = payload
    st.session_state.show_detail_modal = True


def close_detail_modal():
    st.session_state.show_detail_modal = False
    st.session_state.detail_modal_payload = None


def render_detail_modal():
    payload = st.session_state.get("detail_modal_payload")
    if not payload:
        return

    meta = payload.get("meta", {})
    cover_url = payload.get("cover_url") or meta.get("banner_image") or meta.get("cover_large")
    backdrop = cover_url or "https://placehold.co/1200x500/111428/FFFFFF?text=OtakuMood"
    genres = payload.get("genres") or []
    score_value = payload.get("score_value")
    score_badge = "score-none"
    if score_value is not None:
        if score_value >= 90:
            score_badge = "score-high"
        elif score_value >= 80:
            score_badge = "score-mid"
        else:
            score_badge = "score-low"

    watch_links = payload.get("watch_links") or []
    why_text = payload.get("why_text")
    description = meta.get("description") or payload.get("clean_description")
    mood_value = payload.get("mood", "")

    with st.modal("Anime detail", key="detail_modal", max_width=1100):
        st.markdown(
            """
            <style>
            [data-testid='stModal'] {background: rgba(4,6,20,0.85) !important;}
            [data-testid='stModalContent'] {padding: 0 !important;}
            [data-testid='stModal'] button[aria-label='Close'] {display: none;}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class='omd-detail-modal'>
                <div class='omd-detail-hero' style="background-image: url('{backdrop}');">
                    <div class='omd-detail-hero-gradient'></div>
                    <div class='omd-detail-hero-content'>
                        <div class='omd-detail-meta-top'>
                            <p class='omd-detail-eyebrow'>Mood-aligned spotlight</p>
                            <h2>{payload.get('title', 'Untitled')}</h2>
                            <div class='omd-detail-meta-row'>
                                <span class='omd-detail-year'>{payload.get('year', '—')}</span>
                                <span class='omd-detail-score detail-score {score_badge}'>{score_value or '—'}</span>
                            </div>
                            <div class='omd-detail-tags'>
                                {''.join([f"<span>{g}</span>" for g in genres[:5]])}
                            </div>
                            <p class='omd-detail-tagline'>{clean_snippet(description, 220)}</p>
                        </div>
                        <div class='omd-detail-action-row'>
                            <button class='omd-btn-primary'>&#9654; Watch Now</button>
                            <button class='omd-btn-secondary'>&#128190; Save</button>
                            <button class='omd-btn-secondary'>&#128077; Like</button>
                            <button class='omd-btn-secondary'>&#128078; Dislike</button>
                        </div>
                    </div>
                </div>
                <div class='omd-detail-body'>
                    <div class='omd-detail-columns'>
                        <div class='omd-detail-column'>
                            <h4>Synopsis</h4>
                            <p>{description or 'No synopsis available.'}</p>
                            <div class='omd-detail-section'>
                                <h4>Why this matches you</h4>
                                <div class='omd-mood-pill'>{mood_value}</div>
                                <p>{why_text}</p>
                            </div>
                        </div>
                        <div class='omd-detail-column'>
                            <h4>Details</h4>
                            <ul class='omd-detail-list'>
                                <li><strong>Year:</strong> {payload.get('year', '—')}</li>
                                <li><strong>Score:</strong> {score_value or '—'}</li>
                                <li><strong>Episodes:</strong> {payload.get('episodes') or '—'}</li>
                                <li><strong>Duration:</strong> {payload.get('duration') or '—'} min</li>
                            </ul>
                            <div class='omd-detail-section'>
                                <h4>Available on</h4>
                                <div class='omd-watch-links'>
                                    {''.join([f"<a href='{link.get('url')}' target='_blank' rel='noopener' class='omd-watch-link'>{link.get('label', 'Watch')}</a>" for link in watch_links]) or '<span class="omd-watch-placeholder">No platforms listed.</span>'}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-spacer-sm'></div>", unsafe_allow_html=True)
        if st.button("Close detail view", use_container_width=True):
            close_detail_modal()

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

sidebar_icons = {
    "mood": "🎭",
    "filters": "📊",
    "favorites": "❤️",
    "preset_90s": "📼",
    "preset_2000s": "💿",
    "preset_modern": "⚡",
    "reset": "🔄",
}

with st.sidebar:
    toggle_label = "→" if st.session_state.sidebar_collapsed else "Seize shadow-console ←"
    if st.button(toggle_label, key="sidebar_toggle"):
        st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed

    sidebar_width = "80px" if st.session_state.sidebar_collapsed else "400px"
    sidebar_padding = "0.6rem 0.3rem" if st.session_state.sidebar_collapsed else "1.3rem"

    st.markdown(
        f"""
        <style>
        section[data-testid="stSidebar"] {{
            width: {sidebar_width} !important;
            min-width: {sidebar_width} !important;
            transition: width 0.25s ease;
        }}
        section[data-testid="stSidebar"] > div:first-child {{
            width: {sidebar_width} !important;
            padding: {sidebar_padding} !important;
        }}
        .main .block-container {{
            transition: margin-left 0.25s ease;
            margin-left: 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.sidebar_collapsed:
        col = st.container()
        if col.button(f"{sidebar_icons['mood']}", key="mood_icon"):
            st.session_state.show_mood_modal = True
        if col.button(f"{sidebar_icons['filters']}", key="filter_icon"):
            st.session_state.show_filter_modal = True
        if col.button(f"{sidebar_icons['favorites']}", key="fav_icon"):
            st.session_state.show_favorites_modal = True
    else:
        st.title("OtakuMood")
        with st.expander("🎭 Mood Controls", expanded=True):
            st.session_state.manual_mood_value = st.selectbox(
                "Override mood",
                ["(auto)", "happy", "chill", "sad", "angry", "romantic", "adventurous", "nostalgic", "inspiring"],
                index=["(auto)", "happy", "chill", "sad", "angry", "romantic", "adventurous", "nostalgic", "inspiring"].index(st.session_state.manual_mood_value),
                key="manual_mood_select",
            )

        with st.expander("📊 Era & Score", expanded=True):
            st.session_state.year_range_value = st.slider(
                "Year range",
                1980,
                2025,
                st.session_state.year_range_value,
                key="year_slider",
            )
            presets = {
                "90s Classics": (1990, 1999),
                "2000s Era": (2000, 2009),
                "Modern Hits": (2010, 2025),
            }
            preset_cols = st.columns(len(presets))
            for (label, years), col in zip(presets.items(), preset_cols):
                if col.button(label):
                    st.session_state.year_slider = years
                    st.session_state.year_range_value = years
                    st.session_state.trigger_results = True
            era_label = f"You're exploring: {st.session_state.year_range_value[0]}–{st.session_state.year_range_value[1]}"
            st.caption(era_label)

            st.session_state.min_score_value = st.slider(
                "Min score ⭐",
                0,
                100,
                st.session_state.min_score_value,
                key="score_slider",
            )

            if st.button("Reset filters"):
                st.session_state.year_range_value = (1990, 2025)
                st.session_state.min_score_value = 50
                st.session_state.trigger_results = True

# ------------------- Hero + intro -------------------
hero_path = PROJECT_ROOT / "src" / "app" / "heroarea.png"
hero_b64 = load_base64(hero_path)
hero_img_tag = f'<img src="data:image/png;base64,{hero_b64}" alt="OtakuMood logo" />' if hero_b64 else ""

st.markdown(
    f"""
    <section class="hero animated-bg">
        <div class="logo-shell">
            {hero_img_tag}
        </div>
        <div class="hero-text">
            <p class="eyebrow">Not just anime. A mood.</p>
            <h1>OtakuMood</h1>
            <p>Describe the story you want tonight and we'll pull it from the ether.</p>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

placeholder_example = PLACEHOLDERS[len(st.session_state.mood_input) % len(PLACEHOLDERS)]

with st.container():
    lead_col, input_col = st.columns([1.3, 2])
    with lead_col:
        st.subheader("What kind of story do you want to step into tonight?")
        st.caption("Use words, emojis, or vibes. We'll interpret the mood magic for you ✨")
        st.markdown("<div class='section-spacer-sm'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="mood-chip-block">
                <div class="mood-icons">
                    <span>😊 Cozy</span>
                    <span>🔥 Hype</span>
                    <span>🌀 Mind-bend</span>
                    <span>💖 Tender</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with input_col:
        st.markdown("<div class='prompt-composer simple'>", unsafe_allow_html=True)
        user_input = st.text_area(
            "",
            placeholder=placeholder_example,
            key="mood_input",
            label_visibility="collapsed",
            height=150,
            on_change=lambda: trigger_results_view(st.session_state.get("mood_input", ""), update_text=False),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        action_cols = st.columns([3, 1])
        with action_cols[0]:
            st.caption("Describe your mood or let OtakuMood sense it automatically.")
        with action_cols[1]:
            if st.button("Summon vibe ✨", use_container_width=True):
                if not st.session_state.mood_input:
                    st.warning("Tell me a feeling first ✨")
                else:
                    trigger_results_view()

        if user_input:
            st.caption("🪄 Interpreting your vibe in real time…")
# ---------------------------------------------------

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

results_placeholder = st.empty()

if not user_input and not st.session_state.trigger_results:
    with results_placeholder.container():
        st.markdown("<div class='section-spacer-lg'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="empty-state mood-portal">
                <span class="sparkle" aria-hidden="true">✨</span>
                <h3>Summon your vibe</h3>
                <p class="subtitle">Tell me the kind of world you want to disappear into tonight.</p>
                <p class="microcopy">Choose a vibe below or describe your perfect escape.</p>
                <div class="portal-illustration" aria-hidden="true"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-spacer-md'></div>", unsafe_allow_html=True)
        st.markdown("<div class='mood-chip-strip'>", unsafe_allow_html=True)
        chip_cols = st.columns(len(SUGGESTIONS), gap="small")
        for idx, (label, text) in enumerate(SUGGESTIONS):
            col = chip_cols[idx]
            if col.button(label, key=f"chip_{idx}"):
                trigger_results_view(text)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    if st.session_state.trigger_results:
        st.session_state.trigger_results = False

    manual_mood = st.session_state.manual_mood_value
    mood_notice = None
    if manual_mood != "(auto)":
        mood = manual_mood
        mood_notice = ("info", f"Manual mood override: {mood}")
    else:
        if mood_pipe:
            pred = mood_pipe(user_input, top_k=1)[0]
            mood = pred['label']
            mood_notice = ("success", f"Predicted mood: {mood}")
        else:
            mood = "chill"
            mood_notice = ("warning", "Mood model not found, defaulting to 'chill'")

    with st.spinner("Summoning anime energy…"):
        candidates = retrieve(user_input, top_k=25)
        reranked = rerank(candidates, mood)[:10]

        year_min, year_max = st.session_state.year_range_value
        min_score = st.session_state.min_score_value

        filtered = []
        for c in reranked:
            meta = c['meta']
            yr = meta.get('start_year') or 0
            score = meta.get('averageScore') or 0
            if yr >= year_min and yr <= year_max and score >= min_score:
                filtered.append(c)

    with results_placeholder.container():
        if mood_notice:
            level, message = mood_notice
            getattr(st, level)(message)
        st.subheader("Top recommendations")
        if not filtered:
            st.warning("No titles matched your filters — try widening the year range or lowering the score threshold.")
        else:
            cols = st.columns(2, gap="large")
            for i, item in enumerate(filtered[:10]):
                meta = item['meta']
                col = cols[i % 2]
                with col:
                    cover_url = meta.get("cover_large")
                    genres = meta.get('genres') or []
                    title = meta.get('title_english') or meta.get('title_romaji') or "Untitled"
                    score_value = meta.get('averageScore')
                    score_display = score_value if score_value is not None else "—"
                    desc = clean_snippet(meta.get("description"))
                    year_display = meta.get('start_year') or '—'

                    if cover_url:
                        cover_markup = f'<img src="{cover_url}" alt="{title} cover">'
                    else:
                        initials = ''.join([w[0] for w in title.split()[:2]]).upper() or "??"
                        cover_markup = f'<div class="cover-placeholder">{initials}</div>'

                    score_class = "score-low"
                    if score_value is None:
                        score_class = "score-none"
                    elif score_value >= 90:
                        score_class = "score-high"
                    elif score_value >= 80:
                        score_class = "score-mid"

                    genres_html = "".join([f"<span class='tag'>{g}</span>" for g in genres[:4]])

                    vibe_summary = f"Feels like {', '.join(genres[:2])}" if genres else "Signature mood match"
                    why_text = f"{title} blends {', '.join(genres[:2])} energy that echoes your {mood} vibe." if genres else f"{title} mirrors your {mood} mood with its storytelling."

                    st.markdown(f"""
                    <article class="card enhanced">
                        <div class="card-cover">{cover_markup}</div>
                        <div class="card-body">
                            <div class="card-head">
                                <div>
                                    <h3>{title}</h3>
                                    <p class="year">{year_display}</p>
                                </div>
                                <span class="score-badge {score_class}">{score_display}</span>
                            </div>
                            <p class="vibe-summary">{vibe_summary}</p>
                            <p class="card-description">{desc or 'No synopsis available yet.'}</p>
                            <div class="tags">{genres_html}</div>
                            <details class="why-collapse">
                                <summary>Why this matches you</summary>
                                <p>{why_text}</p>
                            </details>
                        </div>
                    </article>
                    """, unsafe_allow_html=True)

                    action_cols = st.columns([1, 1])
                    if action_cols[0].button("Explain", key=f"explain_{i}"):
                        payload = {
                            "meta": meta,
                            "clean_description": clean_snippet(meta.get("description"), limit=2000),
                            "genres": genres,
                            "mood": mood,
                            "why_text": why_text,
                            "cover_url": cover_url,
                            "score_value": score_value,
                            "title": title,
                            "year": year_display,
                            "episodes": meta.get("episodes"),
                            "duration": meta.get("duration"),
                            "watch_links": meta.get("watchLinks", []),
                        }
                        open_detail_modal(payload)
                    if action_cols[1].button("Save", key=f"fav_{i}"):
                        fav_file = Path("data_processed/favorites.json")
                        favs = []
                        if fav_file.exists():
                            favs = json.loads(fav_file.read_text(encoding="utf-8"))
                        favs.append({"meta": meta})
                        fav_file.write_text(json.dumps(favs, ensure_ascii=False, indent=2), encoding="utf-8")
                        st.success("Added to favorites 💾")

def favorites_panel():
    fav_file = Path("data_processed/favorites.json")
    if fav_file.exists():
        favs = json.loads(fav_file.read_text(encoding="utf-8"))
        if favs:
            for f in favs[-5:]:
                meta = f.get("meta", {})
                cover = meta.get("cover_large") or "https://placehold.co/60x80/0f0f1f/FF4DFF?text=☆"
                title = meta.get("title_english") or meta.get("title_romaji") or "Untitled"
                st.markdown(
                    f"""
                    <div class="favorite-chip">
                        <img src="{cover}" alt="{title} cover">
                        <div>
                            <strong>{title}</strong><br/>
                            <span>{meta.get('genres', ['Anime classic'])[0] if meta.get('genres') else 'Anime classic'}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No favorites saved yet.")
    else:
        st.caption("No favorites saved yet.")

if not st.session_state.sidebar_collapsed:
    with st.sidebar.expander("❤️ Favorites shrine", expanded=False):
        favorites_panel()
else:
    if st.session_state.show_mood_modal:
        with st.expander("🎭 Mood Controls", expanded=True):
            st.session_state.manual_mood_value = st.selectbox(
                "Override mood",
                ["(auto)", "happy", "chill", "sad", "angry", "romantic", "adventurous", "nostalgic", "inspiring"],
                index=["(auto)", "happy", "chill", "sad", "angry", "romantic", "adventurous", "nostalgic", "inspiring"].index(st.session_state.manual_mood_value),
                key="manual_mood_modal",
            )
            if st.button("Close mood", key="close_mood"):
                st.session_state.show_mood_modal = False
    if st.session_state.show_filter_modal:
        with st.expander("📊 Era & Score", expanded=True):
            st.session_state.year_range_value = st.slider(
                "Year range",
                1980,
                2025,
                st.session_state.year_range_value,
                key="year_modal",
            )
            st.session_state.min_score_value = st.slider(
                "Min score ⭐",
                0,
                100,
                st.session_state.min_score_value,
                key="score_modal",
            )
            if st.button("Close filters", key="close_filters"):
                st.session_state.show_filter_modal = False
    if st.session_state.show_favorites_modal:
        with st.expander("❤️ Favorites", expanded=True):
            favorites_panel()
            if st.button("Close favorites", key="close_favs"):
                st.session_state.show_favorites_modal = False

st.markdown("<div class='section-spacer-lg'></div>", unsafe_allow_html=True)
with st.container():
    st.markdown("---")
    st.subheader("🎬 Watch the OtakuMood demo")
    if DEMO_VIDEO_PATH:
        video_path = Path(DEMO_VIDEO_PATH)
        if video_path.exists():
            st.video(video_path.read_bytes())
        else:
            st.error(f"Couldn't find demo video at {video_path}")
    else:
        st.info("Set `DEMO_VIDEO_PATH` in your .env file to surface the demo video here.")

    st.markdown("#### 🧭 Quick guide")
    for tip in PROJECT_GUIDE_POINTS:
        st.markdown(f"- {tip}")

if st.session_state.show_detail_modal:
    render_detail_modal()
