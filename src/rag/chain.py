# src/rag/chain.py
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from src.rag.retriever import retrieve
from src.rag.scoring import rerank

PROMPT_TMPL = """
You are OtakuMood, an anime recommender assistant.
User mood: {mood}
User query: {query}
Use the following retrieved context to answer or recommend. Be concise and include why it matches the mood (explicit mapping).
{context}

Answer:
"""

def make_context_text(retrieved):
    parts = []
    for r in retrieved:
        meta = r["meta"]
        t = f"{meta.get('title_english') or meta.get('title_romaji')}: {meta.get('description','')[:300]} Genres: {', '.join(meta.get('genres') or [])}"
        parts.append(t)
    return "\n\n".join(parts)

def run_chain(query, mood, llm_api_key=None):
    # 1. Retrieve candidate anime
    candidates = retrieve(query, top_k=15)
    reranked = rerank(candidates, mood)[:8]
    context = make_context_text(reranked)

    # 2. Build the prompt
    prompt = PROMPT_TMPL.format(mood=mood, query=query, context=context)

    # 3. Initialize Groq LLM
    llm = ChatGroq(
        api_key=llm_api_key,        # pass GROQ_API_KEY from .env
        model="llama3-8b-8192",     # free Groq model
        temperature=0.1
    )

    # 4. Run the chain
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)
