# src/training/synthesize_labels.py
from datasets import Dataset
import random
from pathlib import Path
moods = ["happy","chill","sad","angry","romantic","adventurous","nostalgic","inspiring"]
intents = ["recommendation","factual_qna","compare","filter"]

recommendation_templates = [
  "I'm feeling {mood}. Recommend me an anime please.",
  "I want something {mood} to watch tonight.",
  "Suggest a {mood} anime with good story."
]

qna_templates = [
  "When did {anime} air?",
  "Who is the director of {anime}?",
  "Is {anime} finished airing?"
]

def make_synthetic(n=2000):
    rows=[]
    sample_animes=["Naruto","One Piece","Clannad","Your Lie in April","Haikyuu!!","Spirited Away","Cowboy Bebop"]
    for i in range(n):
        mood=random.choice(moods)
        t=random.choice(recommendation_templates).format(mood=mood)
        rows.append({"text": t, "mood": mood, "intent": "recommendation"})
    for i in range(int(n*0.2)):
        a=random.choice(sample_animes)
        t=random.choice(qna_templates).format(anime=a)
        rows.append({"text": t, "mood": "neutral", "intent": "factual_qna"})
    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=0.15)
    Path("data_processed").mkdir(parents=True, exist_ok=True)
    split["train"].to_parquet("data_processed/train.parquet")
    split["test"].to_parquet("data_processed/test.parquet")
    print("Saved synthetic train/test in data_processed/")
    return split

if __name__=="__main__":
    make_synthetic(2000)
