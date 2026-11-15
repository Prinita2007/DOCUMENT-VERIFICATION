from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.main import app

__all__ = ["app"]
