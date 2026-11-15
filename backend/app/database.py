from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


engine_options: dict[str, object] = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session


def validate_database() -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        inspector = inspect(connection)
        required = {
            "admin_users",
            "students",
            "documents",
            "verification_results",
            "document_history",
            "audit_logs",
        }
        missing = required - set(inspector.get_table_names())
        if missing:
            raise RuntimeError(
                "Database schema is not initialized. Run `python -m alembic upgrade head`; "
                f"missing tables: {', '.join(sorted(missing))}"
            )
