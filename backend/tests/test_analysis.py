import io

import pymupdf
from PIL import Image
from sklearn.ensemble import IsolationForest

from app.services.document_analysis import (
    InvalidDocument,
    _extract_fields,
    detect_document_type,
    preprocess_image,
    tesseract_status,
)
from app.services.ml_anomaly import FEATURE_COLUMNS, detect_anomaly, extract_features


def test_extract_student_fields_and_dates_marks():
    fields = _extract_fields(
        "Student Name: Sample Student\nRegistration No: ABC1234\nDate: 12/06/2024\nMarks: 456/600"
    )
    assert fields["student_name"] == "Sample Student"
    assert fields["registration_number"] == "ABC1234"
    assert "12/06/2024" in fields["dates"]
    assert "456/600" in fields["marks"]


def test_extract_comprehensive_academic_fields_and_validation():
    sample_text = """
    STATE TECHNOLOGICAL UNIVERSITY
    OFFICIAL STATEMENT OF GRADES
    Student Name: Vikram Sharma
    Registration No: 2023CSE0042
    Course: Bachelor of Technology in Computer Science
    Semester: 5th Semester
    Academic Year: 2023-2024
    Date of Issue: 18/06/2023

    CS-501 Operating Systems 100 85 A+
    CS-502 Database Management 100 90 O
    CS-503 Computer Networks 100 78 A

    Total Marks: 253 / 300
    CGPA: 8.01 / 10.0
    Result: PASS
    """
    fields = _extract_fields(sample_text)
    assert fields["student_name"] == "Vikram Sharma"
    assert fields["registration_number"] == "2023CSE0042"
    assert fields["university_name"] == "STATE TECHNOLOGICAL UNIVERSITY"
    assert "Bachelor of Technology" in fields["course_name"]
    assert fields["department"] == "Computer Science"
    assert "5" in fields["semester"]
    assert fields["academic_year"] == "2023-2024"
    assert fields["cgpa"] == "8.01 / 10.0"
    assert fields["result_status"] == "PASS"
    assert fields["issue_date"] == "18/06/2023"
    assert len(fields["subjects"]) == 3
    assert fields["subjects"][0]["name"] == "Operating Systems"
    assert fields["subjects"][0]["obtained_marks"] == 85.0
    assert fields["validation"]["is_marks_consistent"] is True
    assert fields["validation"]["calculated_percentage"] == round((253 / 300) * 100, 2)


def test_tampered_marks_detection_in_fields():
    # Spliced transcript with reported total 299/300 while subjects sum to 253/300
    tampered_text = """
    STATE TECHNOLOGICAL UNIVERSITY
    Student Name: Vikram Sharma
    Registration No: 2023CSE0042
    CS-501 Operating Systems 100 85 A+
    CS-502 Database Management 100 90 O
    CS-503 Computer Networks 100 78 A
    Total Marks: 299 / 300
    """
    fields = _extract_fields(tampered_text)
    assert fields["validation"]["is_marks_consistent"] is False
    assert any("Possible tampering" in a for a in fields["validation"]["anomalies_detected"])


def test_preprocessing_and_pdf_mime_validation():
    image = Image.new("RGB", (400, 600), "white")
    assert preprocess_image(image).mode == "L"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Student Name: Sample Student")
    data = pdf.tobytes()
    pdf.close()
    assert detect_document_type("record.pdf", "application/pdf", data) == (".pdf", "application/pdf")
    try:
        detect_document_type("broken.pdf", "application/pdf", b"%PDF-1.7 broken")
    except InvalidDocument:
        pass
    else:
        raise AssertionError("Malformed PDF was not rejected")


def test_ml_runs_on_real_manually_verified_feature_baseline(signed_in_client):
    from app.database import SessionLocal
    from app.models import Document, Student
    from app.services.ml_anomaly import MINIMUM_BASELINE
    import numpy as np

    with SessionLocal() as db:
        student = Student(full_name="ML Baseline Student")
        db.add(student)
        db.flush()
        for i in range(MINIMUM_BASELINE):
            db.add(
                Document(
                    student_id=student.id,
                    document_type="sslc",
                    stored_filename=f"baseline-{i}.png",
                    original_filename=f"baseline-{i}.png",
                    media_type="image/png",
                    file_size=20_000 + i * 10,
                    sha256=f"{i:064x}",
                    status="verified",
                    features={
                        key: float(value)
                        for key, value in zip(FEATURE_COLUMNS, [10 + i * 0.001, 1000, 1400, 150, 500, 1, 80])
                    },
                )
            )
        db.flush()
        result = detect_anomaly(
            {key: float(value) for key, value in zip(FEATURE_COLUMNS, [10, 1000, 1400, 150, 500, 1, 80])},
            db,
        )
        assert result["status"] in {"outlier", "within_baseline"}
        assert result["prediction"] in {-1, 1}
        assert result["baseline_count"] == MINIMUM_BASELINE


def test_ml_does_not_run_without_verified_baseline(signed_in_client):
    from app.database import SessionLocal
    from app.services.ml_anomaly import FEATURE_COLUMNS, detect_anomaly

    with SessionLocal() as db:
        result = detect_anomaly({key: 1.0 for key in FEATURE_COLUMNS}, db)
    assert result["status"] == "insufficient_baseline"
    assert result["prediction"] is None


def test_tesseract_installation_is_detected_honestly():
    result = tesseract_status()
    assert isinstance(result["available"], bool)
    if not result["available"]:
        assert result["version"] is None
