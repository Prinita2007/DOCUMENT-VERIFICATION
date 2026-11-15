from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    email: Mapped[str | None] = mapped_column(String(254), unique=True)
    registration_number: Mapped[str | None] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    documents: Mapped[list["Document"]] = relationship(back_populates="student")


class AdminUser(Base):
    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(24), default="admin", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    audit_events: Mapped[list["AuditLog"]] = relationship(back_populates="admin")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_sha256", "sha256"),
        Index("ix_documents_student_created", "student_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="RESTRICT"), nullable=False)
    document_type: Mapped[str] = mapped_column(String(48), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    extracted_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    ocr_confidence: Mapped[float | None] = mapped_column(nullable=True)
    extracted_fields: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    features: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    ocr_findings: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    rule_findings: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    ml_findings: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    processing_notes: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    student: Mapped[Student] = relationship(back_populates="documents")
    results: Mapped[list["VerificationResult"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    history: Mapped[list["DocumentHistory"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class VerificationResult(Base):
    __tablename__ = "verification_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    indicators: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    ocr_findings: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    rule_findings: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    ml_findings: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    final_decision: Mapped[str] = mapped_column(String(24), default="pending_review", nullable=False)
    model_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    document: Mapped[Document] = relationship(back_populates="results")


class DocumentHistory(Base):
    __tablename__ = "document_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    action: Mapped[str] = mapped_column(String(48), nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    document: Mapped[Document] = relationship(back_populates="history")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_admin_created", "admin_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    admin_id: Mapped[int | None] = mapped_column(ForeignKey("admin_users.id", ondelete="SET NULL"))
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_type: Mapped[str | None] = mapped_column(String(48))
    target_id: Mapped[str | None] = mapped_column(String(64))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    admin: Mapped[AdminUser | None] = relationship(back_populates="audit_events")
