import io
from pathlib import Path

from PIL import Image, ImageDraw
import pytest
from starlette.testclient import TestClient

from app.database import SessionLocal
from app.models import AdminUser, AuditLog, Document, Student, VerificationResult
from app.security import hash_password


@pytest.fixture
def test_image_bytes():
    image = Image.new("RGB", (600, 800), color=(250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.text((50, 50), "STATE TECHNOLOGICAL UNIVERSITY", fill=(0, 0, 0))
    draw.text((50, 90), "Student Name: Arjun Patel", fill=(0, 0, 0))
    draw.text((50, 120), "Registration No: 2023CSE0099", fill=(0, 0, 0))
    draw.text((50, 150), "Course: Bachelor of Technology in Computer Science", fill=(0, 0, 0))
    draw.text((50, 180), "Semester: 5th Semester", fill=(0, 0, 0))
    draw.text((50, 210), "Date of Issue: 15/06/2023", fill=(0, 0, 0))
    draw.text((50, 260), "CS-501 Operating Systems 100 85 A+", fill=(0, 0, 0))
    draw.text((50, 290), "CS-502 Database Management 100 90 O", fill=(0, 0, 0))
    draw.text((50, 340), "Total Marks: 175 / 200", fill=(0, 0, 0))
    draw.text((50, 370), "CGPA: 8.31 / 10.0", fill=(0, 0, 0))
    draw.text((50, 400), "Result: PASS", fill=(0, 0, 0))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_complete_end_to_end_workflow(client: TestClient, test_image_bytes: bytes):
    # 1. Administrator login
    with SessionLocal() as db:
        admin = AdminUser(
            username="test_admin_e2e",
            password_hash=hash_password("admin-test-password-1234"),
            role="admin",
        )
        db.add(admin)
        db.commit()

    login_resp = client.post(
        "/auth/login",
        json={"username": "test_admin_e2e", "password": "admin-test-password-1234"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Register student
    student_resp = client.post(
        "/students",
        headers=headers,
        json={
            "full_name": "Arjun Patel",
            "registration_number": "2023CSE0099",
            "email": "arjun.patel@test.edu",
        },
    )
    assert student_resp.status_code == 201
    student_id = student_resp.json()["id"]

    # 3. Upload document
    upload_resp = client.post(
        "/documents/upload",
        headers=headers,
        data={"student_id": str(student_id), "document_type": "marks_card"},
        files={"file": ("transcript.png", test_image_bytes, "image/png")},
    )
    assert upload_resp.status_code == 201
    doc_data = upload_resp.json()
    doc_id = doc_data["id"]
    assert doc_data["original_filename"] == "transcript.png"
    assert "latest_result" in doc_data
    assert doc_data["latest_result"]["risk_level"] in {"low", "medium", "high"}

    # 4. Check extracted academic fields
    doc_detail = client.get(f"/documents/{doc_id}", headers=headers).json()
    fields = doc_detail["extracted_fields"]
    assert "student_name" in fields
    assert "registration_number" in fields
    assert "validation" in fields
    assert isinstance(fields["validation"]["is_marks_consistent"], bool)

    # 5. Check local SHAP explanation endpoint
    expl_resp = client.get(f"/documents/{doc_id}/explanation", headers=headers)
    assert expl_resp.status_code == 200
    expl_data = expl_resp.json()
    assert expl_data["document_id"] == doc_id
    assert expl_data["explanation"]["status"] == "available"
    assert "narrative_explanation" in expl_data["explanation"]
    assert len(expl_data["explanation"]["feature_contributions"]) > 0

    # 6. Check ML summary endpoint
    ml_resp = client.get("/ml/summary", headers=headers)
    assert ml_resp.status_code == 200
    assert ml_resp.json()["status"] == "trained"
    assert "logistic_regression" in ml_resp.json()["models"]

    # 7. Check Dashboard stats
    stats_resp = client.get("/dashboard/stats", headers=headers)
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_students"] >= 1
    assert stats["total_documents"] >= 1

    # 8. Administrator review
    review_resp = client.post(
        f"/documents/{doc_id}/review",
        headers=headers,
        json={"decision": "verified", "note": "Verified against institutional registry."},
    )
    assert review_resp.status_code == 200
    assert review_resp.json()["status"] == "verified"

    # 9. Document history check
    history_resp = client.get(f"/documents/{doc_id}/history", headers=headers)
    assert history_resp.status_code == 200
    actions = [h["action"] for h in history_resp.json()]
    assert "uploaded" in actions
    assert "admin_review" in actions

    # 10. Audit log check
    audit_resp = client.get("/admin/audit-logs", headers=headers)
    assert audit_resp.status_code == 200
    events = [a["event_type"] for a in audit_resp.json()]
    assert "document_uploaded" in events
    assert "document_reviewed" in events


def test_invalid_and_malicious_upload_rejection(client: TestClient):
    with SessionLocal() as db:
        admin = AdminUser(
            username="admin_sec_test",
            password_hash=hash_password("admin-test-password-sec-123"),
            role="admin",
        )
        db.add(admin)
        student = Student(full_name="Security Test Student")
        db.add(student)
        db.commit()
        student_id = student.id

    login = client.post(
        "/auth/login",
        json={"username": "admin_sec_test", "password": "admin-test-password-sec-123"},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    # Test 1: Fake executable disguised as PDF
    bad_pdf = client.post(
        "/documents/upload",
        headers=headers,
        data={"student_id": str(student_id), "document_type": "marks_card"},
        files={"file": ("exploit.pdf", b"MZThis is not a PDF", "application/pdf")},
    )
    assert bad_pdf.status_code == 400

    # Test 2: Text file disguised as PNG
    bad_png = client.post(
        "/documents/upload",
        headers=headers,
        data={"student_id": str(student_id), "document_type": "marks_card"},
        files={"file": ("test.png", b"Plain text disguised", "image/png")},
    )
    assert bad_png.status_code == 400

    # Test 3: Unsupported file extension
    bad_ext = client.post(
        "/documents/upload",
        headers=headers,
        data={"student_id": str(student_id), "document_type": "marks_card"},
        files={"file": ("script.exe", b"executable payload", "application/octet-stream")},
    )
    assert bad_ext.status_code in (400, 415)
