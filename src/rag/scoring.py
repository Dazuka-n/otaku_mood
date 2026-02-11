# src/rag/scoring.py
MOOD_TO_GENRES = {
    "happy":["Comedy","Slice of Life","School"],
    "chill":["Slice of Life","Iyashikei","Healing"],
    "sad":["Drama","Psychological","Tragedy"],
    "angry":["Action","Thriller","Revenge"],
    "romantic":["Romance","Shoujo","Josei"],
    "adventurous":["Adventure","Fantasy","Isekai"],
    "nostalgic":["Slice of Life","Classic"],
    "inspiring":["Sports","Drama","Shounen"]
}

def mood_boost_score(candidate_meta, mood):
    base = candidate_meta.get("score", 0.0)
    genres = candidate_meta.get("meta", {}).get("genres", []) or []
    boost = 0.0
    for g in genres:
        if g in MOOD_TO_GENRES.get(mood, []):
            boost += 0.15
    pop = candidate_meta.get("meta", {}).get("popularity") or 0
    pop_norm = min(pop / 100000.0, 1.0)
    final = 0.6*base + 0.3*boost + 0.1*pop_norm
    return final

def rerank(candidates, mood):
    scored=[]
    for c in candidates:
        s = mood_boost_score(c, mood)
        scored.append((s,c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _,c in scored]
