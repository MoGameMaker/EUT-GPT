import os
import sys

APP_NAME = "EUTGPT"

def get_app_dir():
    # Always writable location (works in exe + normal python)
    base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


APP_DIR = get_app_dir()

DB_PATH = os.path.join(APP_DIR, "WikiDump.db")

MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.gguf")

os.makedirs(MODEL_DIR, exist_ok=True)