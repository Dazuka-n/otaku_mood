#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
pip install -r requirements.txt || true
mkdir -p data_raw data_processed models faiss_index
echo "Bootstrap complete. Run make_demo.py to build a demo index, then start the app:"
echo "  python make_demo.py"
echo "  python -m streamlit run src/app/app.py"
