import json
import re
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from app.config import (
    EVALUATION_DIR,
    AUTH_TOKEN_TTL_SECONDS,
    LOGIN_RATE_LIMIT,
    LOGIN_RATE_WINDOW_SECONDS,
    MAX_UPLOAD_SIZE_BYTES,
    UPLOAD_DIR,
)
from app.database import get_db
from app.models import AdminUser, AuditLog, Document, DocumentHistory, Student, VerificationResult
from app.schemas import AdminCreate, AdminLogin, ReviewRequest, StudentCreate, StudentOut, StudentUpdate
from app.security import (
    hash_password,
    issue_token,
    password_needs_upgrade,
    require_admin,
    require_staff,
    verify_password,
)
from app.services.document_analysis import (
    ALLOWED_EXTENSIONS,
    InvalidDocument,
    analyze_document,
    detect_document_type,
    sha256_hex,
)

router = APIRouter()
_login_attempts: OrderedDict[str, list[float]] = OrderedDict()
_login_attempts_lock = Lock()
MAX_RATE_LIMIT_KEYS = 10_000


def _read_result_file(filename: str) -> dict | None:
    path = EVALUATION_DIR / filename
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path.name}.")
    return value


def _write_history(db: Session, document_id: int, action: str, details: dict | None = None) -> None:
    db.add(DocumentHistory(document_id=document_id, action=action, details=details or {}))


def _write_audit(
    db: Session,
    event_type: str,
    request: Request,
    admin_id: int | None = None,
    target_type: str | None = None,
    target_id: int | str | None = None,
    details: dict | None = None,
) -> None:
    db.add(
        AuditLog(
            admin_id=admin_id,
            event_type=event_type,
            target_type=target_type,
            target_id=str(target_id) if target_id is not None else None,
            ip_address=request.client.host if request.client else None,
            details=details or {},
        )
    )


def _is_login_rate_limited(client: str) -> bool:
    now = time.monotonic()
    with _login_attempts_lock:
        for key in list(_login_attempts):
            recent = [stamp for stamp in _login_attempts[key] if now - stamp < LOGIN_RATE_WINDOW_SECONDS]
            if recent:
                _login_attempts[key] = recent
            else:
                _login_attempts.pop(key, None)
        attempts = _login_attempts.get(client, [])
        if len(attempts) >= LOGIN_RATE_LIMIT:
            return True
        if client not in _login_attempts and len(_login_attempts) >= MAX_RATE_LIMIT_KEYS:
            _login_attempts.popitem(last=False)
        _login_attempts[client] = attempts
        _login_attempts.move_to_end(client)
        attempts.append(now)
        return False


def _clear_login_attempts(client: str) -> None:
    with _login_attempts_lock:
        _login_attempts.pop(client, None)


def _result_json(result: VerificationResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "id": result.id,
        "risk_score": result.risk_score,
        "risk_level": result.risk_level,
        "indicators": result.indicators,
        "ocr_findings": result.ocr_findings,
        "rule_findings": result.rule_findings,
        "ml_findings": result.ml_findings,
        "final_decision": result.final_decision,
        "model_notes": result.model_notes,
        "created_at": result.created_at,
    }


def _document_json(document: Document, include_text: bool = False) -> dict:
    latest = max(document.results, key=lambda item: item.id, default=None)
    response = {
        "id": document.id,
        "student_id": document.student_id,
        "student_name": document.student.full_name if document.student else None,
        "student_registration_number": document.student.registration_number if document.student else None,
        "document_type": document.document_type,
        "original_filename": document.original_filename,
        "media_type": document.media_type,
        "file_size": document.file_size,
        "sha256": document.sha256,
        "extracted_fields": document.extracted_fields,
        "ocr_confidence": document.ocr_confidence,
        "ocr_findings": document.ocr_findings,
        "rule_findings": document.rule_findings,
        "ml_findings": document.ml_findings,
        "features": document.features,
        "processing_notes": document.processing_notes,
        "status": document.status,
        "created_at": document.created_at,
        "latest_result": _result_json(latest),
    }
    if include_text:
        response["extracted_text"] = document.extracted_text
    return response


@router.post("/auth/login")
def login(payload: AdminLogin, request: Request, db: Session = Depends(get_db)) -> dict:
    client = request.client.host if request.client else "unknown"
    if _is_login_rate_limited(client):
        raise HTTPException(status_code=429, detail="Too many login attempts; try again in five minutes")
    admin = db.scalar(select(AdminUser).where(AdminUser.username == payload.username))
    if admin is None or not admin.is_active or not verify_password(payload.password, admin.password_hash):
        _write_audit(db, "login_failed", request, details={"reason": "invalid_credentials"})
        db.commit()
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    _clear_login_attempts(client)
    if password_needs_upgrade(admin.password_hash):
        admin.password_hash = hash_password(payload.password)
    _write_audit(db, "login_succeeded", request, admin_id=admin.id, target_type="admin", target_id=admin.id)
    db.commit()
    return {
        "access_token": issue_token(admin),
        "token_type": "bearer",
        "expires_in": AUTH_TOKEN_TTL_SECONDS,
    }


@router.get("/auth/me")
def current_admin(admin: AdminUser = Depends(require_staff)) -> dict:
    return {"id": admin.id, "username": admin.username, "role": admin.role}


@router.post("/students", response_model=StudentOut, status_code=201)
def create_student(
    payload: StudentCreate,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_admin),
) -> Student:
    student = Student(
        full_name=payload.full_name,
        email=payload.email.strip().lower() if payload.email else None,
        registration_number=payload.registration_number.strip() if payload.registration_number else None,
    )
    db.add(student)
    try:
        db.flush()
        _write_audit(db, "student_created", request, admin.id, "student", student.id)
        db.commit()
        db.refresh(student)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Email or registration number is already registered") from exc
    return student


@router.put("/students/{student_id}", response_model=StudentOut)
def update_student(
    student_id: int,
    payload: StudentUpdate,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_admin),
) -> Student:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    values = payload.model_dump(exclude_unset=True)
    if not values:
        raise HTTPException(status_code=422, detail="Provide at least one field to update")
    for key, value in values.items():
        if key == "email" and value == "":
            value = None
        if key == "registration_number" and value == "":
            value = None
        setattr(student, key, value)
    try:
        _write_audit(
            db,
            "student_updated",
            request,
            admin.id,
            "student",
            student.id,
            {"updated_fields": sorted(values)},
        )
        db.commit()
        db.refresh(student)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Email or registration number is already registered") from exc
    return student


@router.delete("/students/{student_id}", status_code=204)
def delete_student(
    student_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_admin),
) -> Response:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    if db.scalar(select(func.count(Document.id)).where(Document.student_id == student_id)):
        raise HTTPException(status_code=409, detail="Student has documents; records with document history cannot be deleted")
    _write_audit(db, "student_deleted", request, admin.id, "student", student_id)
    db.delete(student)
    db.commit()
    return Response(status_code=204)


@router.get("/students", response_model=list[StudentOut])
def list_students(
    request: Request,
    q: str = Query(default="", max_length=120),
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_staff),
) -> list[Student]:
    statement = select(Student).order_by(Student.created_at.desc()).limit(500)
    if q.strip():
        term = f"%{q.strip()}%"
        statement = statement.where(
            (Student.full_name.ilike(term))
            | (Student.registration_number.ilike(term))
            | (Student.email.ilike(term))
        )
    students = list(db.scalars(statement).all())
    _write_audit(db, "students_listed", request, admin.id, details={"search": bool(q.strip())})
    db.commit()
    return students


@router.post("/documents/upload", status_code=201)
async def upload_document(
    request: Request,
    student_id: int = Form(...),
    document_type: str = Form(..., pattern="^(sslc|puc|degree|transfer|other|marks_card|transcript)$"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_admin),
) -> dict:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    original = Path(file.filename or "upload").name
    original = re.sub(r"[\x00-\x1f\x7f]", "", original)[:255] or "upload"
    extension = Path(original).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only PDF, JPG, JPEG, and PNG files are accepted")
    chunks: list[bytes] = []
    total = 0
    try:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds the configured upload-size limit")
            chunks.append(chunk)
    finally:
        await file.close()
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    try:
        extension, media_type = detect_document_type(original, file.content_type or "", data)
    except InvalidDocument as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    digest = sha256_hex(data)
    duplicate_count = db.scalar(select(func.count(Document.id)).where(Document.sha256 == digest)) or 0
    stored_name = f"{uuid.uuid4().hex}{extension}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = UPLOAD_DIR / stored_name
    document = Document(
        student_id=student.id,
        document_type=document_type,
        stored_filename=stored_name,
        original_filename=original,
        media_type=media_type,
        file_size=len(data),
        sha256=digest,
    )
    try:
        analysis = analyze_document(
            data, extension, document, student, duplicate_count, db
        )
        with stored_path.open("xb") as destination:
            destination.write(data)
        db.add(document)
        db.flush()
        result = VerificationResult(
            document_id=document.id,
            risk_score=analysis["risk_score"],
            risk_level=analysis["risk_level"],
            indicators=analysis["indicators"],
            ocr_findings=analysis["ocr_findings"],
            rule_findings=analysis["rule_findings"],
            ml_findings=analysis["ml_findings"],
            model_notes=analysis["model_notes"],
        )
        document.results.append(result)
        _write_history(db, document.id, "uploaded", {"admin_id": admin.id, "sha256": digest})
        _write_audit(db, "document_uploaded", request, admin.id, "document", document.id)
        db.commit()
        db.refresh(document)
    except Exception:
        db.rollback()
        if stored_path.exists() and stored_path.parent.resolve() == UPLOAD_DIR.resolve():
            stored_path.unlink()
        raise
    return _document_json(document, include_text=True)


@router.get("/documents")
def list_documents(
    request: Request,
    q: str = Query(default="", max_length=120),
    status_filter: str | None = Query(default=None, alias="status", pattern="^(pending|verified|suspicious)$"),
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_staff),
) -> list[dict]:
    statement = (
        select(Document)
        .options(joinedload(Document.student), joinedload(Document.results))
        .order_by(Document.created_at.desc())
        .limit(500)
    )
    if status_filter:
        statement = statement.where(Document.status == status_filter)
    if q.strip():
        term = f"%{q.strip()}%"
        statement = statement.join(Document.student).where(
            (Document.original_filename.ilike(term))
            | (Student.full_name.ilike(term))
            | (Student.registration_number.ilike(term))
        )
    documents = [_document_json(item) for item in db.scalars(statement).unique().all()]
    _write_audit(db, "documents_listed", request, admin.id, details={"search": bool(q.strip()), "status": status_filter})
    db.commit()
    return documents


@router.get("/documents/{document_id}")
def get_document(
    document_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_staff),
) -> dict:
    document = db.scalar(
        select(Document)
        .options(joinedload(Document.student), joinedload(Document.results))
        .where(Document.id == document_id)
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    _write_audit(db, "document_viewed", request, admin.id, "document", document_id)
    db.commit()
    return _document_json(document, include_text=True)


@router.post("/documents/{document_id}/verify")
def verify_document(
    document_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_staff),
) -> dict:
    document = db.scalar(
        select(Document).options(joinedload(Document.student), joinedload(Document.results)).where(Document.id == document_id)
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    path = UPLOAD_DIR / document.stored_filename
    if not path.is_file() or path.parent.resolve() != UPLOAD_DIR.resolve():
        raise HTTPException(status_code=404, detail="Stored document file is unavailable")
    data = path.read_bytes()
    duplicate_count = max(
        0,
        (db.scalar(select(func.count(Document.id)).where(Document.sha256 == document.sha256)) or 0) - 1,
    )
    extension = Path(document.stored_filename).suffix.lower()
    analysis = analyze_document(
        data, extension, document, document.student, duplicate_count, db
    )
    result = VerificationResult(
        document_id=document.id,
        risk_score=analysis["risk_score"],
        risk_level=analysis["risk_level"],
        indicators=analysis["indicators"],
        ocr_findings=analysis["ocr_findings"],
        rule_findings=analysis["rule_findings"],
        ml_findings=analysis["ml_findings"],
        model_notes=analysis["model_notes"],
    )
    document.results.append(result)
    _write_history(
        db,
        document.id,
        "verification_rerun",
        {"admin_id": admin.id, "risk_level": analysis["risk_level"]},
    )
    _write_audit(db, "document_verification_rerun", request, admin.id, "document", document.id)
    db.commit()
    db.refresh(document)
    return _document_json(document, include_text=True)


@router.post("/documents/{document_id}/review")
def review_document(
    document_id: int,
    payload: ReviewRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_staff),
) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    document.status = payload.decision
    latest = max(document.results, key=lambda item: item.id, default=None)
    if latest is not None:
        latest.final_decision = payload.decision
    _write_history(
        db,
        document.id,
        "admin_review",
        {"admin_id": admin.id, "decision": payload.decision, "note": payload.note.strip()},
    )
    _write_audit(
        db,
        "document_reviewed",
        request,
        admin.id,
        "document",
        document.id,
        {"decision": payload.decision},
    )
    db.commit()
    return {"id": document.id, "status": document.status}


@router.get("/documents/{document_id}/file")
def download_document(
    document_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_staff),
) -> FileResponse:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    path = UPLOAD_DIR / document.stored_filename
    if not path.is_file() or path.resolve().parent != UPLOAD_DIR.resolve():
        raise HTTPException(status_code=404, detail="Stored document file is unavailable")
    _write_audit(db, "document_downloaded", request, admin.id, "document", document.id)
    db.commit()
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=document.original_filename,
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"},
    )


@router.get("/documents/{document_id}/history")
def document_history(
    document_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_staff),
) -> list[dict]:
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    _write_audit(db, "document_history_viewed", request, admin.id, "document", document_id)
    db.commit()
    entries = db.scalars(
        select(DocumentHistory)
        .where(DocumentHistory.document_id == document_id)
        .order_by(DocumentHistory.created_at.desc())
    ).all()
    return [
        {"id": item.id, "action": item.action, "details": item.details, "created_at": item.created_at}
        for item in entries
    ]


@router.get("/verification-results")
def verification_results(
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_staff),
) -> list[dict]:
    rows = db.scalars(
        select(VerificationResult).order_by(VerificationResult.created_at.desc()).limit(500)
    ).all()
    return [
        {
            **(_result_json(row) or {}),
            "document_id": row.document_id,
        }
        for row in rows
    ]


@router.get("/ml/summary")
def ml_summary(_: AdminUser = Depends(require_staff)) -> dict:
    comparison = _read_result_file("model_comparison.json")
    cnn = _read_result_file("cnn_results.json")
    if comparison is None and cnn is None:
        return {
            "status": "dataset_required",
            "message": "No dataset-trained evaluation artifacts are available. Supply an approved labeled dataset and run the documented training commands.",
            "models": [],
            "cnn": None,
            "shap": None,
        }
    return {
        "status": "trained",
        "models": comparison.get("models", {}) if comparison else {},
        "selected_supervised_model": comparison.get("selected_supervised_model") if comparison else None,
        "row_counts": comparison.get("row_counts") if comparison else None,
        "manifest_sha256": comparison.get("manifest_sha256") if comparison else None,
        "cnn": cnn,
        "shap": comparison.get("shap") if comparison else None,
        "limitations": [
            "Reported metrics are from the locally trained dataset and its held-out source-group split.",
            "Predictions and explanations are indicators for manual review, not proof of fraud.",
        ],
    }


@router.get("/dashboard/stats")
def dashboard_stats(
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_staff),
) -> dict:
    total_students = db.scalar(select(func.count(Student.id))) or 0
    total_documents = db.scalar(select(func.count(Document.id))) or 0
    verified = db.scalar(select(func.count(Document.id)).where(Document.status == "verified")) or 0
    suspicious = db.scalar(select(func.count(Document.id)).where(Document.status == "suspicious")) or 0
    pending = db.scalar(select(func.count(Document.id)).where(Document.status == "pending")) or 0
    _write_audit(db, "dashboard_viewed", request, admin.id)
    db.commit()
    return {
        "total_students": total_students,
        "total_documents": total_documents,
        "verified_documents": verified,
        "pending_documents": pending,
        "suspicious_documents": suspicious,
        "generated_at": datetime.now(timezone.utc),
    }


@router.get("/admin/audit-logs")
def audit_logs(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_admin),
) -> list[dict]:
    rows = db.scalars(
        select(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    ).all()
    return [
        {
            "id": row.id,
            "admin_id": row.admin_id,
            "event_type": row.event_type,
            "target_type": row.target_type,
            "target_id": row.target_id,
            "ip_address": row.ip_address,
            "details": row.details,
            "created_at": row.created_at,
        }
        for row in rows
    ]


@router.get("/documents/{document_id}/explanation")
def document_explanation(
    document_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_staff),
) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    from ml.explainability import explain_document_features

    explanation = explain_document_features(document.features or {})
    _write_audit(db, "document_explanation_viewed", request, admin.id, "document", document_id)
    db.commit()
    return {
        "document_id": document.id,
        "filename": document.original_filename,
        "features": document.features,
        "explanation": explanation,
    }


@router.get("/admin/users")
def list_admin_users(
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> list[dict]:
    users = db.scalars(select(AdminUser).order_by(AdminUser.created_at.desc())).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at,
        }
        for u in users
    ]


@router.post("/admin/users", status_code=201)
def create_admin_user(
    payload: AdminCreate,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_admin),
) -> dict:
    from app.services.admin_accounts import register_admin

    try:
        new_user = register_admin(db, payload.username, payload.password, payload.role)
        _write_audit(
            db,
            "admin_user_created",
            request,
            admin.id,
            "admin_user",
            new_user.id,
            {"username": new_user.username, "role": new_user.role},
        )
        db.commit()
        return {
            "id": new_user.id,
            "username": new_user.username,
            "role": new_user.role,
            "is_active": new_user.is_active,
            "created_at": new_user.created_at,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/admin/users/{user_id}")
def delete_admin_user(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(require_admin),
) -> dict:
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own active administrator account")
    target = db.get(AdminUser, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(target)
    _write_audit(db, "admin_user_deleted", request, admin.id, "admin_user", user_id, {"username": target.username})
    db.commit()
    return {"status": "deleted", "id": user_id}
