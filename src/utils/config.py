# src/utils/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

def load_env():
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    return dict(os.environ)
