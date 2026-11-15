"""Create application tables and bring forward the existing SQLite demo schema.

Revision ID: 20261114_0001
Revises:
Create Date: 2026-11-14
"""
from alembic import op
import sqlalchemy as sa

from app.database import Base
from app import models  # noqa: F401

revision = "20261114_0001"
down_revision = None
branch_labels = None
depends_on = None


def _add_missing_column(table: str, column: sa.Column) -> None:
    inspector = sa.inspect(op.get_bind())
    if table in inspector.get_table_names():
        existing = {item["name"] for item in inspector.get_columns(table)}
        if column.name not in existing:
            op.add_column(table, column)


def upgrade() -> None:
    bind = op.get_bind()
    # create_all is check-first and non-destructive; it also upgrades the older
    # SQLite demo database without removing its existing student/document rows.
    Base.metadata.create_all(bind=bind, checkfirst=True)

    _add_missing_column(
        "admin_users",
        sa.Column("role", sa.String(length=24), nullable=False, server_default="admin"),
    )
    _add_missing_column("documents", sa.Column("ocr_confidence", sa.Float(), nullable=True))
    _add_missing_column(
        "documents", sa.Column("ocr_findings", sa.JSON(), nullable=False, server_default=sa.text("'[]'"))
    )
    _add_missing_column(
        "documents", sa.Column("rule_findings", sa.JSON(), nullable=False, server_default=sa.text("'[]'"))
    )
    _add_missing_column(
        "documents", sa.Column("ml_findings", sa.JSON(), nullable=False, server_default=sa.text("'[]'"))
    )
    _add_missing_column(
        "verification_results",
        sa.Column("ocr_findings", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    _add_missing_column(
        "verification_results",
        sa.Column("rule_findings", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    _add_missing_column(
        "verification_results",
        sa.Column("ml_findings", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    _add_missing_column(
        "verification_results",
        sa.Column(
            "final_decision",
            sa.String(length=24),
            nullable=False,
            server_default=sa.text("'pending_review'"),
        ),
    )


def downgrade() -> None:
    raise RuntimeError(
        "This initial migration intentionally has no destructive downgrade. "
        "Back up the database and create a reviewed migration before changing schema."
    )
