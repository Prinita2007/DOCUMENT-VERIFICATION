from __future__ import annotations

from pathlib import Path
import shutil
import sys
import uuid

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_backend_dir = _project_root / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from sqlalchemy import select
from app.config import UPLOAD_DIR
from app.database import SessionLocal, validate_database
from app.models import AdminUser, AuditLog, Document, DocumentHistory, Student, VerificationResult
from app.services.admin_accounts import register_admin
from app.services.document_analysis import analyze_document, sha256_hex


def seed() -> None:
    validate_database()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        # 1. Demo Administrator
        demo_admin = db.scalar(select(AdminUser).where(AdminUser.username == "demo_admin"))
        if not demo_admin:
            demo_admin = register_admin(
                db,
                username="demo_admin",
                password="AdminPassword@2026",
                role="admin",
            )
            print(f"Created demo administrator: {demo_admin.username}")
        else:
            print(f"Demo administrator already exists: {demo_admin.username}")

        # 2. Demo Students
        students_info = [
            ("Rohan Patel", "2023CSE0042", "rohan.patel@demo.edu"),
            ("Neha Sharma", "2023ECE0018", "neha.sharma@demo.edu"),
            ("Arjun Nair", "2023BCA0009", "arjun.nair@demo.edu"),
        ]
        created_students = []
        for name, reg_no, email in students_info:
            existing = db.scalar(select(Student).where(Student.registration_number == reg_no))
            if not existing:
                student = Student(full_name=name, registration_number=reg_no, email=email)
                db.add(student)
                db.flush()
                created_students.append(student)
                print(f"Created demo student: {name} ({reg_no})")
            else:
                created_students.append(existing)

        # 3. Demo Documents
        synthetic_dir = _project_root / "datasets" / "synthetic_benchmark" / "images"
        if synthetic_dir.is_dir() and created_students:
            sample_files = [
                ("doc_src000_var0_authentic.png", created_students[0], "marks_card"),
                ("doc_src001_var0_tampered.png", created_students[1], "transcript"),
            ]
            for filename, student, doc_type in sample_files:
                src_path = synthetic_dir / filename
                if not src_path.is_file():
                    continue

                content = src_path.read_bytes()
                digest = sha256_hex(content)

                # Check if already added
                existing_doc = db.scalar(select(Document).where(Document.sha256 == digest))
                if existing_doc:
                    continue

                stored_name = f"{uuid.uuid4().hex}.png"
                dest_path = UPLOAD_DIR / stored_name
                shutil.copyfile(src_path, dest_path)

                doc = Document(
                    student_id=student.id,
                    document_type=doc_type,
                    stored_filename=stored_name,
                    original_filename=f"[DEMO] {filename}",
                    media_type="image/png",
                    file_size=len(content),
                    sha256=digest,
                    status="pending",
                )
                db.add(doc)
                db.flush()

                # Run real analysis pipeline
                analysis = analyze_document(
                    data=content,
                    extension=".png",
                    document=doc,
                    student=student,
                    duplicate_count=0,
                    db=db,
                )

                # Save VerificationResult
                vres = VerificationResult(
                    document_id=doc.id,
                    risk_score=analysis["risk_score"],
                    risk_level=analysis["risk_level"],
                    indicators=analysis["indicators"],
                    ocr_findings=analysis["ocr_findings"],
                    rule_findings=analysis["rule_findings"],
                    ml_findings=analysis["ml_findings"],
                    final_decision="pending_review",
                    model_notes=analysis["model_notes"],
                )
                db.add(vres)

                # Add history
                db.add(
                    DocumentHistory(
                        document_id=doc.id,
                        action="uploaded_and_analyzed",
                        details={"demo_record": True, "analysis_status": "complete"},
                    )
                )

                # Add audit log
                db.add(
                    AuditLog(
                        admin_id=demo_admin.id,
                        event_type="document_uploaded_demo",
                        target_type="document",
                        target_id=str(doc.id),
                        details={"filename": doc.original_filename, "risk_level": analysis["risk_level"]},
                    )
                )

                print(f"Created demo document and analysis for {student.full_name}: Risk {analysis['risk_score']}/100 ({analysis['risk_level']})")

        db.commit()
        print("Demo database initialization completed successfully.")


if __name__ == "__main__":
    seed()
