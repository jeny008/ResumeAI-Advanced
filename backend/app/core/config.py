import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA / "uploads")))
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", str(DATA / "qdrant")))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
