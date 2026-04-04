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
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq

from src.utils.config import load_env
from src.rag.retriever import retrieve
from src.rag.scoring import rerank

load_dotenv()
ENV = load_env()

st.set_page_config(page_title="OtakuMood", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    z-index: 1 !important;
}
.omd-detail-modal-overlay {
    z-index: 9999 !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
}
</style>
""", unsafe_allow_html=True)

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
    st.session_state.min_score_value = 70
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
# last_query holds the text at the moment the user pressed submit —
# separate from mood_input so filter changes don't re-trigger mood detection.
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "detected_mood" not in st.session_state:
    st.session_state.detected_mood = None
if "query_run" not in st.session_state:
    st.session_state.query_run = False

def trigger_results_view(value: str | None = None, update_text: bool = False):
    if value is None:
        value = st.session_state.get("mood_input", "")
    # update_text is intentionally ignored — writing to a widget key after
    # instantiation raises StreamlitAPIException. last_query holds the query.
    if value.strip():
        st.session_state.last_query = value.strip()
        st.session_state.trigger_results = True
        st.session_state.query_run = True
        st.session_state.detected_mood = None  # reset so mood is re-detected for new query

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
    return text[:limit].rsplit(" ", 1)[0] + "..."


local_css("src/app/style.css")
# -------------------------------------------------------

def open_detail_modal(payload: dict):
    st.session_state.detail_modal_payload = payload
    st.session_state.show_detail_modal = True
    st.rerun()


def close_detail_modal():
    st.session_state.show_detail_modal = False
    st.session_state.detail_modal_payload = None


def render_detail_modal():
    payload = st.session_state.get("detail_modal_payload")
    if not payload:
        return

    meta = payload.get("meta", {})
    cover_url = payload.get("cover_url") or meta.get("cover_url")
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

    # ── STEP 1: Native close button rendered FIRST so Streamlit tracks it
    # before any HTML content is injected. JS below will mark it and hide it.
    if st.button("✕ Close", key="close_detail_modal_btn"):
        close_detail_modal()
        st.rerun()

    # ── STEP 2: Escape key closes the modal via the native button above
    components.html("""
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            var btn = window.parent.document.querySelector('button[data-omd-close="true"]');
            if (btn) btn.click();
        }
    });
    </script>
    """, height=0)

    # ── STEP 3: HTML overlay (× triggers triggerOmdClose which clicks the
    # native button via data-omd-close attribute set by the wiring script)
    overlay_html = f"""
    <div class='omd-detail-modal-overlay' onclick="if(event.target === this) window.triggerOmdClose(event);">
        <div class='omd-detail-modal' onclick="event.stopPropagation()">
            <button class='omd-detail-close' type='button' aria-label="Close detail" onclick="window.triggerOmdClose(event)"><span>&times;</span></button>
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
                            <li><strong>Duration:</strong> {payload.get('duration_mins') or '—'} min</li>
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
    </div>
    <script>
        window.triggerOmdClose = window.triggerOmdClose || function(event) {{
            if (event) {{
                event.stopPropagation();
                event.preventDefault();
            }}
            var btn = window.parent.document.querySelector('button[data-omd-close="true"]');
            if (btn) {{
                btn.click();
            }} else {{
                window.parent.postMessage({{type:'omd-close-modal'}}, '*');
            }}
        }};
    </script>
    """
    st.markdown(overlay_html, unsafe_allow_html=True)

    # ── STEP 4: Wire the native button — mark it, visually hide it, and
    # handle the postMessage fallback in case the attribute isn't set yet.
    components.html(
        """
        <script>
        (function() {
            function wire() {
                var btns = window.parent.document.querySelectorAll('button');
                btns.forEach(function(btn) {
                    if (btn.innerText.trim() === '\u2715 Close') {
                        btn.setAttribute('data-omd-close', 'true');
                        btn.style.position = 'absolute';
                        btn.style.width = '1px';
                        btn.style.height = '1px';
                        btn.style.opacity = '0';
                        btn.style.pointerEvents = 'none';
                        btn.style.overflow = 'hidden';
                        btn.style.clip = 'rect(0,0,0,0)';
                    }
                });
            }
            wire();
            setTimeout(wire, 120);
            window.addEventListener('message', function(event) {
                if (event.data && event.data.type === 'omd-close-modal') {
                    var btn = window.parent.document.querySelector('button[data-omd-close="true"]');
                    if (btn) btn.click();
                }
            });
        })();
        </script>
        """,
        height=0,
    )

# Groq client — reads from st.secrets first, falls back to env
def _get_groq_key():
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")

client = Groq(api_key=_get_groq_key())

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
            _mood_opts = ["(auto)", "happy", "sad", "chill", "excited", "anxious", "romantic", "dark", "nostalgic"]
            _mood_idx = _mood_opts.index(st.session_state.manual_mood_value) if st.session_state.manual_mood_value in _mood_opts else 0
            st.session_state.manual_mood_value = st.selectbox(
                "Override mood",
                _mood_opts,
                index=_mood_idx,
                key="manual_mood_select",
            )

        with st.expander("📊 Era & Score", expanded=True):
            # Preset buttons rendered BEFORE the slider so the staging key
            # can be read and applied as the slider's default value.
            presets = {
                "90s Classics": (1990, 1999),
                "2000s Era": (2000, 2009),
                "Modern Hits": (2010, 2025),
            }
            preset_cols = st.columns(len(presets))
            for (label, years), col in zip(presets.items(), preset_cols):
                if col.button(label):
                    st.session_state.year_preset = years
                    st.rerun()

            if st.button("Reset filters"):
                st.session_state.year_preset = (1990, 2025)
                st.session_state.min_score_value = 70
                st.rerun()

            # Apply staged preset before slider instantiation, then clear it.
            # pop() returns None when key is absent; fall back to current range.
            default_years = st.session_state.pop("year_preset", None) or st.session_state.year_range_value

            st.session_state.year_range_value = st.slider(
                "Year range",
                1980,
                2025,
                default_years,
                key="year_slider",
            )
            era_label = f"You're exploring: {st.session_state.year_range_value[0]}–{st.session_state.year_range_value[1]}"
            st.caption(era_label)

            st.session_state.min_score_value = st.slider(
                "Min score ⭐",
                0,
                100,
                st.session_state.min_score_value,
                key="score_slider",
            )

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
    with input_col:
        st.markdown("<div class='prompt-composer simple'>", unsafe_allow_html=True)
        st.text_area(
            "Describe your vibe",
            placeholder=placeholder_example,
            key="mood_input",
            label_visibility="collapsed",
            height=150,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        action_cols = st.columns([3, 1])
        with action_cols[0]:
            st.caption("Describe your mood or let OtakuMood sense it automatically.")
        with action_cols[1]:
            if st.button("Summon vibe ✨", use_container_width=True):
                query = st.session_state.get("mood_input", "").strip()
                if not query:
                    st.warning("Tell me a feeling first ✨")
                else:
                    trigger_results_view(query)

# ---------------------------------------------------

def run_chain_groq(query, mood):
    """LLM explanation using Groq."""
    prompt = f"""The user feels '{mood}' and wrote: '{query}'.
Explain briefly why the recommended anime matches their mood and preferences."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Groq error: {e})"

def detect_mood(text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
                    "Classify the following text into exactly one mood label.\n"
                    "Choose only from: happy, sad, chill, excited, anxious, romantic, dark, nostalgic\n"
                    f'Text: "{text}"\n'
                    "Reply with only the single mood label, nothing else."
                ),
            }],
            max_tokens=10,
            temperature=0,
        )
        mood = resp.choices[0].message.content.strip().lower()
        valid = {"happy", "sad", "chill", "excited", "anxious", "romantic", "dark", "nostalgic"}
        return mood if mood in valid else "chill"
    except Exception as e:
        st.warning(f"Mood detection unavailable ({e}), defaulting to 'chill'")
        return "chill"


results_placeholder = st.empty()

if not st.session_state.last_query:
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
                trigger_results_view(text, update_text=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.session_state.trigger_results = False
    query = st.session_state.last_query

    manual_mood = st.session_state.manual_mood_value
    mood_notice = None
    if manual_mood != "(auto)":
        mood = manual_mood
        st.session_state.detected_mood = mood
        mood_notice = ("info", f"Manual mood override: {mood}")
    elif st.session_state.detected_mood:
        # reuse cached mood — avoids Groq call on every filter slider change
        mood = st.session_state.detected_mood
        mood_notice = ("success", f"Detected mood: {mood}")
    else:
        with st.spinner("Summoning your vibe... ✨"):
            mood = detect_mood(query)
            st.session_state.detected_mood = mood
        mood_notice = ("success", f"Detected mood: {mood}")

    with st.spinner("Finding your anime... ✨"):
        candidates = retrieve(query, top_k=25)
        reranked = rerank(candidates, mood)[:10]

        year_min, year_max = st.session_state.year_range_value
        min_score = st.session_state.min_score_value

        filtered = []
        seen_ids = set()
        for c in reranked:
            meta = c['meta']
            aid = meta.get('id')
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            yr = meta.get('year') or 0
            score = meta.get('score') or 0
            if yr >= year_min and yr <= year_max and score >= min_score:
                filtered.append(c)

    components.html(
        "<script>window.parent.scrollTo({top: 600, behavior: 'smooth'});</script>",
        height=0,
    )

    with results_placeholder.container():
        if mood_notice:
            level, message = mood_notice
            getattr(st, level)(message)
        st.subheader("Top recommendations")
        if not filtered:
            st.info("No anime found. Try adjusting your filters or rewording your mood.")
        else:
            cols = st.columns(2, gap="large")
            for i, item in enumerate(filtered[:10]):
                meta = item['meta']
                col = cols[i % 2]
                with col:
                    cover_url = meta.get("cover_url")
                    genres_raw = meta.get('genres') or ''
                    genres = [g.strip() for g in genres_raw.split(',') if g.strip()]
                    _t = meta.get('title') or "Untitled"
                    title = _t[0].upper() + _t[1:] if _t else "Untitled"
                    score_value = meta.get('score')
                    score_display = score_value if score_value is not None else "—"
                    desc = clean_snippet(meta.get("description"))
                    year_display = meta.get('year') or '—'

                    if cover_url and str(cover_url) != 'nan':
                        cover_markup = f'<img src="{cover_url}" alt="{title} cover">'
                    else:
                        cover_markup = f"<div class='cover-placeholder'>{title[0].upper() if title else '?'}</div>"

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
                            "year": meta.get("year") or "—",
                            "episodes": meta.get("episodes"),
                            "duration_mins": meta.get("duration_mins"),
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
                cover = meta.get("cover_url") or "https://placehold.co/60x80/0f0f1f/FF4DFF?text=☆"
                title = meta.get("title") or "Untitled"
                genres_raw = meta.get("genres") or ""
                if isinstance(genres_raw, list):
                    first_genre = genres_raw[0] if genres_raw else "Anime classic"
                else:
                    first_genre = genres_raw.split(",")[0].strip() if genres_raw else "Anime classic"
                st.markdown(
                    f"""
                    <div class="favorite-chip">
                        <img src="{cover}" alt="{title} cover">
                        <div>
                            <strong>{title}</strong><br/>
                            <span>{first_genre}</span>
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
            _mood_opts_m = ["(auto)", "happy", "sad", "chill", "excited", "anxious", "romantic", "dark", "nostalgic"]
            _mood_idx_m = _mood_opts_m.index(st.session_state.manual_mood_value) if st.session_state.manual_mood_value in _mood_opts_m else 0
            st.session_state.manual_mood_value = st.selectbox(
                "Override mood",
                _mood_opts_m,
                index=_mood_idx_m,
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
    if DEMO_VIDEO_PATH:
        video_path = Path(DEMO_VIDEO_PATH)
        if video_path.exists():
            st.markdown("---")
            st.markdown("<h3 style='margin-bottom:0.5rem'>🎬 Watch the OtakuMood demo</h3>", unsafe_allow_html=True)
            st.video(video_path.read_bytes())

    if not st.session_state.get("query_run"):
        st.markdown("---")
        st.markdown("#### 🧭 Quick guide")
        for tip in PROJECT_GUIDE_POINTS:
            st.markdown(f"- {tip}")

if st.session_state.show_detail_modal:
    render_detail_modal()
