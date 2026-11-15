from pathlib import Path
import sys

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import DATABASE_URL  # noqa: E402


def main() -> None:
    url = make_url(DATABASE_URL)
    if not url.drivername.startswith("postgresql"):
        raise SystemExit("DATABASE_URL must point to PostgreSQL to create the requested database.")
    if not url.username:
        raise SystemExit("Set DATABASE_URL with a PostgreSQL application role before continuing.")
    try:
        with psycopg.connect(
            host=url.host or "localhost",
            port=url.port or 5432,
            user=url.username,
            password=url.password,
            dbname="postgres",
            connect_timeout=5,
            sslmode=url.query.get("sslmode", "prefer"),
        ) as connection:
            connection.autocommit = True
            exists = connection.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (url.database or "document_verification",),
            ).fetchone()
            if exists:
                print("Database already exists; no change made.")
                return
            connection.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(url.database or "document_verification")
                )
            )
            print("Database created. Existing databases were not modified.")
    except psycopg.OperationalError as exc:
        raise SystemExit(
            "Could not authenticate/connect to PostgreSQL. Verify the local DATABASE_URL "
            "and role in .env or pgAdmin. No authentication configuration was changed."
        ) from exc
    except psycopg.errors.InsufficientPrivilege as exc:
        raise SystemExit(
            "The configured PostgreSQL role cannot create databases. Ask a PostgreSQL administrator "
            "to create document_verification and grant access to this role."
        ) from exc


if __name__ == "__main__":
    main()
