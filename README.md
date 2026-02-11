# OtakuMood

OtakuMood is a mood-driven anime recommender + chatbot that combines a fine-tuned BERT classifier with a RAG pipeline (FAISS + sentence-transformers) and LangChain for LLM orchestration. The frontend is built with Streamlit for quick demos.

## Quickstart (demo)
1. Create virtualenv and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. Create demo index and catalog:
   ```
   python make_demo.py
   ```
3. Run app:
   ```
   python -m streamlit run src/app/app.py
   ```

See `src/` for scripts to fetch data, preprocess, build embeddings, train classifier, and run the app.
