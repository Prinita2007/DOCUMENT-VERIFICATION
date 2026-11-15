import io

from PIL import Image, ImageDraw
import pytest
from sqlalchemy import select

from app.database import SessionLocal
from app.models import AdminUser, AuditLog, Document, DocumentHistory, Student, VerificationResult
from app.security import hash_password, verify_password
from app.services.admin_accounts import register_admin


def png_document() -> bytes:
    image = Image.new("RGB", (1100, 1500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 120), "SAMPLE EDUCATION BOARD", fill="black")
    draw.text((80, 200), "Student Name: Sample Demo Student", fill="black")
    draw.text((80, 260), "Registration No: DEMO-12345", fill="black")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_argon2_password_hash_and_verification():
    encoded = hash_password("a-long-password-for-tests")
    assert encoded.startswith("$argon2id$")
    assert verify_password("a-long-password-for-tests", encoded)
    assert not verify_password("wrong-password", encoded)


def test_admin_account_creation_validates_password_role_and_duplicate(client):
    with SessionLocal() as db:
        created = register_admin(db, "  first-admin  ", "a-strong-password-here")
        assert created.username == "first-admin"
        assert created.role == "admin"
        assert created.password_hash.startswith("$argon2id$")
        with pytest.raises(ValueError, match="already exists"):
            register_admin(db, "first-admin", "another-strong-password")
        with pytest.raises(ValueError, match="12 characters"):
            register_admin(db, "second-admin", "short")
        with pytest.raises(ValueError, match="Role must"):
            register_admin(db, "second-admin", "a-strong-password-here", "owner")


def test_authentication_and_role_authorization(client):
    assert client.get("/dashboard/stats").status_code == 401
    assert client.get("/ml/summary").status_code == 401
    assert client.post("/auth/login", json={"username": "missing", "password": "invalid"}).status_code == 401
    with SessionLocal() as db:
        db.add(
            AdminUser(
                username="reviewer",
                password_hash=hash_password("reviewer-password-long-enough"),
                role="reviewer",
            )
        )
        db.commit()
    login = client.post(
        "/auth/login",
        json={"username": "reviewer", "password": "reviewer-password-long-enough"},
    )
    assert login.status_code == 200
    client.headers.update({"Authorization": f"Bearer {login.json()['access_token']}"})
    assert client.get("/auth/me").json()["role"] == "reviewer"
    assert client.post("/students", json={"full_name": "Restricted Student"}).status_code == 403
    assert client.get("/students").status_code == 200
    assert client.get("/ml/summary").json()["status"] in {"dataset_required", "trained"}
    with SessionLocal() as db:
        assert db.scalar(select(AuditLog).where(AuditLog.event_type == "login_failed")) is not None
        assert db.scalar(select(AuditLog).where(AuditLog.event_type == "login_succeeded")) is not None


def test_student_crud_and_protected_delete(signed_in_client):
    created = signed_in_client.post(
        "/students",
        json={
            "full_name": "Sample Student",
            "email": "sample@example.edu",
            "registration_number": "SAMPLE-001",
        },
    )
    assert created.status_code == 201
    student_id = created.json()["id"]

    updated = signed_in_client.put(
        f"/students/{student_id}",
        json={"full_name": "Updated Sample Student", "email": "updated@example.edu"},
    )
    assert updated.status_code == 200
    assert updated.json()["full_name"] == "Updated Sample Student"
    assert updated.json()["email"] == "updated@example.edu"
    assert signed_in_client.put(f"/students/{student_id}", json={}).status_code == 422
    assert signed_in_client.delete(f"/students/{student_id}").status_code == 204
    assert signed_in_client.delete(f"/students/{student_id}").status_code == 404


def test_complete_document_workflow_and_persistence(signed_in_client):
    response = signed_in_client.post(
        "/students",
        json={
            "full_name": "Sample Demo Student",
            "registration_number": "DEMO-12345",
            "email": "demo@example.edu",
        },
    )
    assert response.status_code == 201, response.text
    student_id = response.json()["id"]

    content = png_document()
    uploads = []
    for _ in range(2):
        uploaded = signed_in_client.post(
            "/documents/upload",
            data={"student_id": str(student_id), "document_type": "sslc"},
            files={"file": ("sample.png", content, "image/png")},
        )
        assert uploaded.status_code == 201, uploaded.text
        uploads.append(uploaded.json())

    duplicate = uploads[1]
    assert duplicate["latest_result"]["risk_level"] in {"low", "medium", "high"}
    assert duplicate["latest_result"]["risk_score"] >= 60
    assert any("Exact file duplicate" in reason for reason in duplicate["rule_findings"])
    if not duplicate["extracted_text"]:
        assert any("Tesseract OCR engine" in reason for reason in duplicate["ocr_findings"])
    assert "ocr_confidence" in duplicate

    document_id = duplicate["id"]
    assert signed_in_client.delete(f"/students/{student_id}").status_code == 409
    details = signed_in_client.get(f"/documents/{document_id}")
    assert details.status_code == 200
    assert details.json()["extracted_fields"]["student_name"] is None or isinstance(
        details.json()["extracted_fields"]["student_name"], str
    )

    verified = signed_in_client.post(f"/documents/{document_id}/verify")
    assert verified.status_code == 200
    assert verified.json()["latest_result"]["ml_findings"]

    review = signed_in_client.post(
        f"/documents/{document_id}/review",
        json={"decision": "verified", "note": "Reviewed in automated test"},
    )
    assert review.status_code == 200
    history = signed_in_client.get(f"/documents/{document_id}/history")
    assert history.status_code == 200
    assert any(row["action"] == "admin_review" for row in history.json())

    stats = signed_in_client.get("/dashboard/stats")
    assert stats.status_code == 200
    assert stats.json()["total_students"] == 1
    assert stats.json()["total_documents"] == 2
    assert stats.json()["verified_documents"] == 1

    download = signed_in_client.get(f"/documents/{document_id}/file")
    assert download.status_code == 200
    assert download.content == content
    assert download.headers["x-content-type-options"] == "nosniff"

    with SessionLocal() as db:
        assert db.get(Student, student_id) is not None
        assert db.get(Document, document_id) is not None
        assert db.scalar(select(VerificationResult).where(VerificationResult.document_id == document_id)) is not None
        assert db.scalar(select(DocumentHistory).where(DocumentHistory.document_id == document_id)) is not None
        assert db.scalar(select(AuditLog).where(AuditLog.event_type == "document_uploaded")) is not None


def test_upload_rejects_bad_extension_and_malformed_pdf(signed_in_client):
    student = signed_in_client.post("/students", json={"full_name": "Demo Student"})
    assert student.status_code == 201
    student_id = student.json()["id"]

    bad_extension = signed_in_client.post(
        "/documents/upload",
        data={"student_id": str(student_id), "document_type": "other"},
        files={"file": ("notes.txt", b"not a document", "text/plain")},
    )
    assert bad_extension.status_code == 415

    malformed_pdf = signed_in_client.post(
        "/documents/upload",
        data={"student_id": str(student_id), "document_type": "other"},
        files={"file": ("broken.pdf", b"%PDF-1.7 this is not a PDF", "application/pdf")},
    )
    assert malformed_pdf.status_code == 415


def test_invalid_mime_and_unauthorized_file_access(signed_in_client):
    student = signed_in_client.post("/students", json={"full_name": "Demo Student"}).json()
    invalid_image = signed_in_client.post(
        "/documents/upload",
        data={"student_id": str(student["id"]), "document_type": "other"},
        files={"file": ("fake.png", b"not png", "image/png")},
    )
    assert invalid_image.status_code == 415

    signed_in_client.headers.pop("Authorization")
    assert signed_in_client.get("/documents/1/file").status_code == 401
