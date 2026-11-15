from pathlib import Path
import sys

from alembic import command
from alembic.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    config = Config(str(ROOT / "alembic.ini"))
    command.upgrade(config, "head")
    print("Database migrations applied.")


if __name__ == "__main__":
    main()
