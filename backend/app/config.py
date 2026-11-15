import os
from pathlib import Path
import secrets
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

EVALUATION_DIR = PROJECT_ROOT / "results" / "evaluation"
load_dotenv(PROJECT_ROOT / ".env")


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return parsed


_default_sqlite = f"sqlite:///{(PROJECT_ROOT / 'database' / 'document_verification.db').resolve().as_posix()}"
_raw_db_url = os.getenv("DATABASE_URL") or _default_sqlite
if _raw_db_url.startswith("sqlite:///") and not _raw_db_url.startswith("sqlite:////") and not (len(_raw_db_url) > 11 and _raw_db_url[11] == ":"):
    _rel_path = _raw_db_url[10:]
    DATABASE_URL = f"sqlite:///{(PROJECT_ROOT / _rel_path).resolve().as_posix()}"
else:
    DATABASE_URL = _raw_db_url

_upload_dir_env = os.getenv("UPLOAD_DIR")
if _upload_dir_env:
    _up = Path(_upload_dir_env)
    UPLOAD_DIR = _up if _up.is_absolute() else (PROJECT_ROOT / _up).resolve()
else:
    UPLOAD_DIR = PROJECT_ROOT / "uploaded_documents"

MAX_UPLOAD_SIZE_BYTES = _positive_int("MAX_UPLOAD_SIZE_MB", 10) * 1024 * 1024
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
JWT_SECRET = os.getenv("JWT_SECRET") or secrets.token_urlsafe(48)
APP_ENV = os.getenv("APP_ENV", "development").lower()
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
AUTH_TOKEN_TTL_SECONDS = _positive_int("AUTH_TOKEN_TTL_SECONDS", 8 * 60 * 60)
LOGIN_RATE_LIMIT = _positive_int("LOGIN_RATE_LIMIT", 5)
LOGIN_RATE_WINDOW_SECONDS = _positive_int("LOGIN_RATE_WINDOW_SECONDS", 300)

if APP_ENV == "production" and len(JWT_SECRET) < 32:
    raise RuntimeError("Set a random JWT_SECRET of at least 32 characters in production")
