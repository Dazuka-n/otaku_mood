# src/rag/scoring.py
MOOD_TO_GENRES = {
    "happy":    ["Comedy", "Slice of Life", "School"],
    "sad":      ["Drama", "Psychological", "Tragedy"],
    "chill":    ["Slice of Life", "Iyashikei", "Healing"],
    "excited":  ["Action", "Adventure", "Sports"],
    "anxious":  ["Psychological", "Thriller", "Horror"],
    "romantic": ["Romance", "Shoujo", "Josei"],
    "dark":     ["Psychological", "Horror", "Thriller"],
    "nostalgic":["Slice of Life", "Drama"],
    # legacy labels kept for manual override compatibility
    "angry":       ["Action", "Thriller"],
    "adventurous": ["Adventure", "Fantasy", "Isekai"],
    "inspiring":   ["Sports", "Drama", "Shounen"],
}


def mood_boost_score(candidate, mood):
    base = candidate.get("score", 0.0)
    meta = candidate.get("meta", {})
    genres_raw = meta.get("genres") or ""
    genres = [g.strip() for g in genres_raw.split(",")] if genres_raw else []
    boost = sum(0.15 for g in genres if g in MOOD_TO_GENRES.get(mood, []))
    pop = meta.get("popularity") or 0
    pop_norm = min(pop / 100000.0, 1.0)
    return 0.6 * base + 0.3 * boost + 0.1 * pop_norm


def rerank(candidates, mood):
    scored = [(mood_boost_score(c, mood), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]
