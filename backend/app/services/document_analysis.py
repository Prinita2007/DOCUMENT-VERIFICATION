import hashlib
import io
import math
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf
import pytesseract
from PIL import Image
from pytesseract import Output
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import TESSERACT_CMD
from app.models import Document, Student
from app.services.ml_anomaly import detect_anomaly, extract_features
from ml.cnn_inference import predict_document_image
from ml.inference import predict_classical_models

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_PDF_PAGES = 5
MAX_IMAGE_PIXELS = 25_000_000
MAX_RENDER_PIXELS = 4_000_000


class InvalidDocument(ValueError):
    pass


def tesseract_status() -> dict[str, Any]:
    configured = TESSERACT_CMD or shutil.which("tesseract")
    if not configured:
        return {"available": False, "executable": None, "version": None}
    if TESSERACT_CMD and not Path(TESSERACT_CMD).is_file():
        return {"available": False, "executable": TESSERACT_CMD, "version": None}
    try:
        pytesseract.pytesseract.tesseract_cmd = configured
        version = str(pytesseract.get_tesseract_version())
        return {"available": True, "executable": configured, "version": version}
    except (pytesseract.TesseractNotFoundError, OSError):
        return {"available": False, "executable": configured, "version": None}


def detect_document_type(filename: str, content_type: str, data: bytes) -> tuple[str, str]:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidDocument("Only PDF, JPG, JPEG, and PNG files are accepted.")
    if ext == ".pdf":
        if not data.startswith(b"%PDF-") or content_type not in {"application/pdf", "application/octet-stream"}:
            raise InvalidDocument("The file does not appear to be a valid PDF.")
        try:
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                if pdf.needs_pass or pdf.page_count < 1:
                    raise InvalidDocument("Encrypted PDFs and PDFs with no pages are not supported.")
        except InvalidDocument:
            raise
        except (pymupdf.FileDataError, RuntimeError) as exc:
            raise InvalidDocument("The PDF could not be decoded safely.") from exc
        return ext, "application/pdf"
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise InvalidDocument("Image dimensions exceed the safe processing limit.")
            image.verify()
            detected = image.format
            if detected not in {"JPEG", "PNG"}:
                raise InvalidDocument("Only JPEG and PNG image contents are accepted.")
            if (ext in {".jpg", ".jpeg"} and detected != "JPEG") or (ext == ".png" and detected != "PNG"):
                raise InvalidDocument("File extension does not match the image contents.")
            expected_mime = "image/jpeg" if detected == "JPEG" else "image/png"
            if content_type not in {expected_mime, "application/octet-stream"}:
                raise InvalidDocument("The provided content type does not match the image.")
            return ext, expected_mime
    except InvalidDocument:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidDocument("The image could not be decoded safely.") from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _render_pdf_page(page: pymupdf.Page, requested_scale: float = 2.0) -> pymupdf.Pixmap:
    rect = page.rect
    scale = min(
        requested_scale,
        4096 / max(rect.width, rect.height, 1),
        math.sqrt(MAX_RENDER_PIXELS / max(rect.width * rect.height, 1)),
    )
    return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)


def _load_first_image(data: bytes, extension: str) -> tuple[Image.Image, int, str, list[str]]:
    notes: list[str] = []
    if extension == ".pdf":
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if not pdf.page_count:
                raise InvalidDocument("The PDF has no pages.")
            if pdf.page_count > MAX_PDF_PAGES:
                notes.append(f"Only the first {MAX_PDF_PAGES} PDF pages are analyzed.")
            pixmap = _render_pdf_page(pdf[0], requested_scale=1.2)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            metadata = pdf.metadata or {}
            software = " ".join(str(metadata.get(key, "")) for key in ("creator", "producer")).lower()
            return image, min(pdf.page_count, MAX_PDF_PAGES), software, notes
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise InvalidDocument("Image dimensions exceed the safe processing limit.")
        image = source.convert("RGB")
        software = str(source.getexif().get(305, "")).lower()
        return image, 1, software, notes


def _quality_analysis(image: Image.Image, pages: int, software: str, notes: list[str]) -> tuple[dict, list[str]]:
    findings = list(notes)
    if any(name in software for name in ("photoshop", "gimp", "canva", "illustrator", "affinity")):
        findings.append("File metadata names image-editing software; metadata alone cannot establish tampering.")
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    height, width = gray.shape
    if width < 500 or height < 350:
        findings.append("Low image resolution may reduce OCR reliability.")
    if blur < 35:
        findings.append("Image appears blurry; review the original document.")
    contrast = float(gray.std())
    return {
        "width": int(width),
        "height": int(height),
        "blur": round(blur, 3),
        "contrast": round(contrast, 3),
        "pages": int(pages),
    }, findings


def deskew_and_orient_image(image: Image.Image) -> Image.Image:
    """Detect image orientation using Tesseract OSD and minor skew via OpenCV."""
    try:
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        osd = pytesseract.image_to_osd(image)
        rotate_match = re.search(r"Rotate:\s*(\d+)", osd)
        if rotate_match:
            angle = int(rotate_match.group(1))
            if angle in (90, 180, 270):
                image = image.rotate(-angle, expand=True)
    except Exception:
        pass

    try:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if 0.5 <= abs(angle) <= 15.0:
                (h, w) = rgb.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                image = Image.fromarray(rotated)
    except Exception:
        pass

    return image


def preprocess_image(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return Image.fromarray(thresholded)


def _ocr_image(image: Image.Image) -> tuple[str, float | None]:
    oriented = deskew_and_orient_image(image)
    processed = preprocess_image(oriented)
    data = pytesseract.image_to_data(processed, output_type=Output.DICT, config="--oem 3 --psm 6")
    lines: list[str] = []
    current_key: tuple[int, int, int] | None = None
    current_words: list[str] = []
    confidences: list[float] = []
    for index, (word, raw_confidence) in enumerate(zip(data["text"], data["conf"])):
        text = word.strip()
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        if current_key is not None and key != current_key and current_words:
            lines.append(" ".join(current_words))
            current_words = []
        current_key = key
        try:
            confidence = float(raw_confidence)
        except (ValueError, TypeError):
            continue
        if text and confidence >= 0:
            current_words.append(text)
            confidences.append(confidence)
    if current_words:
        lines.append(" ".join(current_words))
    text = "\n".join(lines).strip()
    confidence = round(float(np.mean(confidences)), 2) if confidences else None
    return text, confidence


def _extract_text(data: bytes, extension: str) -> tuple[str, float | None, str, list[str]]:
    notes: list[str] = []
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    try:
        if extension == ".pdf":
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                native_text = "\n".join(page.get_text("text") for page in pdf[:MAX_PDF_PAGES]).strip()
                if len(native_text) >= 30:
                    return native_text[:100_000], None, "embedded_pdf_text", notes
                images = []
                for page in pdf[:MAX_PDF_PAGES]:
                    pix = _render_pdf_page(page)
                    with Image.open(io.BytesIO(pix.tobytes("png"))) as rendered:
                        images.append(rendered.convert("RGB"))
                extracted = [_ocr_image(image) for image in images]
                text = "\n".join(item[0] for item in extracted).strip()
                confidences = [item[1] for item in extracted if item[1] is not None]
                confidence = round(float(np.mean(confidences)), 2) if confidences else None
                method = "tesseract_preprocessed"
        else:
            with Image.open(io.BytesIO(data)) as image:
                text, confidence = _ocr_image(image.convert("RGB"))
            method = "tesseract_preprocessed"
        if not text:
            notes.append("No readable text was extracted; manual review is required.")
        if confidence is not None and confidence < 50:
            notes.append("OCR confidence is low; extracted fields may be inaccurate and need manual review.")
        return text[:100_000], confidence, method, notes
    except pytesseract.pytesseract.TesseractNotFoundError:
        return "", None, "unavailable", ["Tesseract OCR engine is not installed/configured; manual review is required."]
    except pytesseract.TesseractError as exc:
        return "", None, "error", [f"Tesseract reported an OCR error ({exc.status}); manual review is required."]
    except (pymupdf.FileDataError, pymupdf.FileNotFoundError, OSError, RuntimeError) as exc:
        return "", None, "error", [f"OCR could not process this document ({type(exc).__name__}); manual review is required."]


def _extract_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "student_name": None,
        "registration_number": None,
        "institution_name": None,
        "university_name": None,
        "course_name": None,
        "department": None,
        "semester": None,
        "academic_year": None,
        "cgpa": None,
        "sgpa": None,
        "percentage": None,
        "issue_date": None,
        "result_status": None,
        "dates": [],
        "marks": [],
        "subjects": [],
        "validation": {
            "is_marks_consistent": None,
            "calculated_percentage": None,
            "calculated_cgpa": None,
            "anomalies_detected": [],
        },
        "field_confidences": {},
    }
    if not text:
        return fields

    # 1. Student Name
    name = re.search(
        r"(?im)^\s*(?:student\s+name|candidate(?:\'s)?\s+name|name\s+of\s+(?:the\s+)?candidate|name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{1,80})",
        text,
    )
    if not name:
        name = re.search(
            r"(?im)\b(?:this\s+is\s+to\s+certify\s+that|certified\s+that)\s+([A-Za-z][A-Za-z .'-]{2,60})",
            text,
        )
    if name:
        val = name.group(1).strip()
        val = re.split(r"(?i)\s+(?:registration|reg|roll|usn|course|father|mother|dob)\b", val)[0].strip()
        fields["student_name"] = val
        fields["field_confidences"]["student_name"] = 92.0

    # 2. Registration / Roll Number
    registration = re.search(
        r"(?im)^\s*(?:registration\s*(?:no|number)?|reg\s*\.?\s*no|roll\s*(?:no|number)|usn|hall\s*ticket\s*(?:no|number)?|enrollment\s*(?:no|number)?|student\s*id)\s*[:#\-]\s*([A-Za-z0-9/-]{3,32})",
        text,
    )
    if not registration:
        registration = re.search(
            r"(?im)\b(?:USN|REG\s*NO|ROLL\s*NO|ENROLLMENT\s*NO)\s*[:#\-]?\s*([A-Za-z0-9/-]{4,24})\b",
            text,
        )
    if registration:
        fields["registration_number"] = registration.group(1).strip()
        fields["field_confidences"]["registration_number"] = 95.0

    # 3. University & Institution Name
    university_patterns = [
        r"(?im)\b([A-Za-z\s&.,'-]+(?:UNIVERSITY|TECHNOLOGICAL UNIVERSITY|DEEMED UNIVERSITY|CENTRAL UNIVERSITY|STATE UNIVERSITY))\b",
        r"(?im)\b(NATIONAL\s+INSTITUTE\s+OF\s+TECHNOLOGY(?:\s+[A-Za-z]+)?)\b",
        r"(?im)\b(INDIAN\s+INSTITUTE\s+OF\s+TECHNOLOGY(?:\s+[A-Za-z]+)?)\b",
        r"(?im)\b([A-Za-z\s&.,'-]+(?:BOARD\s+OF\s+(?:SECONDARY|TECHNICAL|HIGHER)?\s*EDUCATION))\b",
    ]
    for pat in university_patterns:
        m = re.search(pat, text)
        if m:
            fields["university_name"] = m.group(1).strip()
            fields["field_confidences"]["university_name"] = 90.0
            break

    institution = re.search(
        r"(?im)^\s*(?:institution|college|school|academy|institute)\s*(?:name)?\s*[:\-]\s*([A-Za-z0-9 .,&'()-]{3,100})",
        text,
    )
    if institution:
        fields["institution_name"] = institution.group(1).strip()
        fields["field_confidences"]["institution_name"] = 88.0
    elif fields.get("university_name"):
        fields["institution_name"] = fields["university_name"]
    else:
        for line in text.splitlines()[:10]:
            line_clean = line.strip()
            if 6 <= len(line_clean) <= 100:
                if re.search(
                    r"\b(?:University|Institute\s+of\s+Technology|College\s+of\s+Engineering|Degree\s+College|Polytechnic|Public\s+School|Pre-University|Education\s+Board|Academy\s+of\s+Sciences)\b",
                    line_clean,
                    re.I,
                ):
                    fields["institution_name"] = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9.)]+$", "", line_clean)
                    fields["field_confidences"]["institution_name"] = 80.0
                    break

    # 4. Course & Department
    course = re.search(
        r"(?im)^\s*(?:course|degree|programme?|branch|stream|examination)\s*(?:name)?\s*[:\-]\s*([A-Za-z0-9 .,&'()/-]{2,80})",
        text,
    )
    if not course:
        course = re.search(
            r"(?im)\b(Bachelor\s+of\s+[A-Za-z ]+(?:in\s+[A-Za-z ]+)?|Master\s+of\s+[A-Za-z ]+(?:in\s+[A-Za-z ]+)?|B\.?Tech(?:nology)?(?:\s+in\s+[A-Za-z ]+)?|B\.?E\.?(?:\s+in\s+[A-Za-z ]+)?|B\.?Sc\.?(?:\s+in\s+[A-Za-z ]+)?|B\.?C\.?A\.?|B\.?Com\.?|B\.?B\.?A\.?|M\.?Tech(?:\s+in\s+[A-Za-z ]+)?|M\.?C\.?A\.?|M\.?B\.?A\.?|M\.?Sc\.?(?:\s+in\s+[A-Za-z ]+)?|Diploma\s+in\s+[A-Za-z ]+|SSLC|PUC|Higher\s+Secondary|Senior\s+Secondary|Secondary\s+School)\b",
            text,
        )
    if course:
        fields["course_name"] = course.group(1).strip()
        fields["field_confidences"]["course_name"] = 90.0

    dept = re.search(
        r"(?im)^\s*(?:department(?:\s+of)?|dept\.?)\s*[:\-]\s*([A-Za-z0-9 .,&'()/-]{2,80})",
        text,
    )
    if dept:
        fields["department"] = dept.group(1).strip()
        fields["field_confidences"]["department"] = 85.0
    elif fields.get("course_name") and " in " in fields["course_name"]:
        fields["department"] = fields["course_name"].split(" in ", 1)[1].strip()

    # 5. Semester & Academic Year
    semester = re.search(
        r"(?im)^\s*(?:semester|sem|term)\s*[:\-]?\s*([0-9IVX]+(?:st|nd|rd|th)?|\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth)\b)",
        text,
    )
    if not semester:
        semester = re.search(
            r"(?im)\b([1-8](?:st|nd|rd|th)?\s*Sem(?:ester)?|Sem(?:ester)?\s*[-:#]?\s*([0-9IVX]+))\b",
            text,
        )
    if semester:
        val = semester.group(1) or (semester.group(2) if semester.lastindex and semester.lastindex >= 2 else None) or semester.group(0)
        fields["semester"] = val.strip()
        fields["field_confidences"]["semester"] = 90.0

    acad_yr = re.search(
        r"(?im)\b(?:academic\s+year|acad\s+year|session|batch)\s*[:\-]?\s*(\d{4}\s*[-/]\s*(?:\d{4}|\d{2})|\d{4})\b",
        text,
    )
    if acad_yr:
        fields["academic_year"] = acad_yr.group(1).strip()
        fields["field_confidences"]["academic_year"] = 85.0

    # 6. CGPA, SGPA & Percentage
    cgpa_m = re.search(
        r"(?im)\b(?:cgpa|c\.g\.p\.a\.?)\s*[:=\-]?\s*(\d(?:\.\d{1,3})?(?:\s*/\s*(?:10|4)(?:\.0)?)?)",
        text,
    )
    if cgpa_m:
        fields["cgpa"] = cgpa_m.group(1).strip()
        fields["field_confidences"]["cgpa"] = 92.0

    sgpa_m = re.search(
        r"(?im)\b(?:sgpa|s\.g\.p\.a\.?|gpa|g\.p\.a\.?)\s*[:=\-]?\s*(\d(?:\.\d{1,3})?(?:\s*/\s*(?:10|4)(?:\.0)?)?)",
        text,
    )
    if sgpa_m:
        fields["sgpa"] = sgpa_m.group(1).strip()
        fields["field_confidences"]["sgpa"] = 90.0

    pct_m = re.search(
        r"(?im)\b(?:percentage|aggregate(?:\s+percentage)?|marks\s+percentage)\s*[:=\-]?\s*(\d{1,3}(?:\.\d{1,2})?\s*%)",
        text,
    )
    if pct_m:
        fields["percentage"] = pct_m.group(1).strip()
        fields["field_confidences"]["percentage"] = 92.0

    # 7. Result Status
    result_m = re.search(
        r"(?im)\b(?:result|status|final\s+result|grade\s+status)\s*[:\-]?\s*(PASS|PASSED|FAIL|FAILED|FIRST\s+CLASS(?:\s+WITH\s+DISTINCTION)?|SECOND\s+CLASS|PROMOTED|DISTINCTION)\b",
        text,
    )
    if result_m:
        fields["result_status"] = result_m.group(1).strip().upper()
        fields["field_confidences"]["result_status"] = 95.0

    # 8. Issue Date and Other Dates
    dates = list(
        dict.fromkeys(
            re.findall(
                r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
                text,
                re.I,
            )
        )
    )[:10]
    fields["dates"] = dates

    issue_date = re.search(
        r"(?im)(?:date\s+of\s+issue|issued\s+(?:on|date)|issue\s+date|date\s+of\s+declaration|declaration\s+date|dated)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})",
        text,
    )
    if issue_date:
        fields["issue_date"] = issue_date.group(1).strip()
        fields["field_confidences"]["issue_date"] = 92.0
    elif dates:
        fields["issue_date"] = dates[0]

    # 9. Raw Marks Patterns
    marks = list(
        dict.fromkeys(re.findall(r"\b\d{1,4}\s*(?:/|out of)\s*\d{1,4}\b", text, re.I))
    )[:20]
    total_marks = re.search(
        r"(?im)(?:total(?:\s+marks)?|aggregate|grand\s+total|marks\s+obtained)\s*[:\-]?\s*(\d{1,4}\s*(?:/|out of)\s*\d{1,4})",
        text,
    )
    if total_marks and total_marks.group(1) not in marks:
        marks.insert(0, total_marks.group(1))
    fields["marks"] = marks

    # 10. Structured Subject Table Parsing
    subject_pattern = re.compile(
        r"^(?:([A-Za-z0-9.]+[-_][0-9A-Za-z]+|[A-Z]{2,5}\d{2,4})\s+)?([A-Za-z0-9 &,/()-]{3,50}?)\s+(\d{2,4})\s+(\d{1,4})(?:\s+([A-O][+]?|[A-F]|PASS|FAIL))?$",
        re.I,
    )
    subjects: list[dict[str, Any]] = []
    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean or "total" in line_clean.lower() or "course code" in line_clean.lower():
            continue
        match = subject_pattern.match(line_clean)
        if match:
            code, title, max_m, obt_m, grade = match.groups()
            try:
                max_val = float(max_m)
                obt_val = float(obt_m)
            except ValueError:
                continue
            subjects.append({
                "code": code.strip() if code else None,
                "name": title.strip(),
                "max_marks": max_val,
                "obtained_marks": obt_val,
                "grade": grade.strip() if grade else None,
            })
    fields["subjects"] = subjects

    # 11. Mathematical Validation & Academic Anomaly Detection
    anomalies: list[str] = []
    total_obtained_calc = 0.0
    total_max_calc = 0.0
    for subj in subjects:
        obt = subj.get("obtained_marks")
        mx = subj.get("max_marks")
        if obt is not None and mx is not None:
            if obt > mx:
                anomalies.append(f"Subject '{subj['name']}': marks obtained ({obt}) exceeds maximum marks ({mx}).")
            total_obtained_calc += obt
            total_max_calc += mx

    if total_max_calc > 0:
        calc_pct = round((total_obtained_calc / total_max_calc) * 100, 2)
        calc_cgpa = round((total_obtained_calc / total_max_calc) * 9.5, 2)
        fields["validation"]["calculated_percentage"] = calc_pct
        fields["validation"]["calculated_cgpa"] = calc_cgpa

    if total_marks and total_max_calc > 0:
        m_parts = re.split(r"/|out of", total_marks.group(1), flags=re.I)
        if len(m_parts) == 2:
            try:
                rep_obt = float(m_parts[0].strip())
                if abs(rep_obt - total_obtained_calc) > 2:
                    anomalies.append(
                        f"Reported total marks ({rep_obt:.0f}) does not match sum of subject marks ({total_obtained_calc:.0f}). Possible tampering."
                    )
            except ValueError:
                pass

    fields["validation"]["anomalies_detected"] = anomalies
    fields["validation"]["is_marks_consistent"] = len(anomalies) == 0

    return fields



def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def analyze_document(
    data: bytes,
    extension: str,
    document: Document,
    student: Student,
    duplicate_count: int,
    db: Session,
) -> dict[str, Any]:
    image, pages, software, page_notes = _load_first_image(data, extension)
    quality, quality_findings = _quality_analysis(image, pages, software, page_notes)
    extracted_text, confidence, extraction_method, ocr_findings = _extract_text(data, extension)
    fields = _extract_fields(extracted_text)
    document.extracted_text = extracted_text
    document.ocr_confidence = confidence
    document.extracted_fields = fields
    features = extract_features(document.file_size, quality, len(extracted_text), confidence)
    document.features = features

    rule_findings: list[str] = []
    if duplicate_count:
        rule_findings.append(f"Exact file duplicate detected in {duplicate_count} previously uploaded document(s).")
    candidate_name = fields.get("student_name")
    registration = fields.get("registration_number")
    if candidate_name and _normalize(candidate_name) != _normalize(student.full_name):
        rule_findings.append("OCR-extracted name does not match the registered student name; verify OCR and the source document.")
    if registration and student.registration_number and _normalize(registration) != _normalize(student.registration_number):
        rule_findings.append("OCR-extracted registration number does not match the student record.")

    for anomaly in fields.get("validation", {}).get("anomalies_detected", []):
        rule_findings.append(f"Academic record anomaly: {anomaly}")

    prior_documents = db.scalars(
        select(Document).where(
            Document.student_id == student.id,
            Document.id != document.id,
        )
    ).all()
    for previous in prior_documents:
        previous_fields = previous.extracted_fields or {}
        previous_name = previous_fields.get("student_name")
        if candidate_name and previous_name and _normalize(candidate_name) != _normalize(previous_name):
            rule_findings.append("OCR-extracted name differs from a prior student document; verify source records.")
            break
    for previous in prior_documents:
        previous_registration = (previous.extracted_fields or {}).get("registration_number")
        if registration and previous_registration and _normalize(registration) != _normalize(previous_registration):
            rule_findings.append("OCR-extracted registration number differs from a prior student document.")
            break

    ml_prediction = detect_anomaly(features, db)
    ml_findings = [ml_prediction["message"]]
    supervised_predictions = predict_classical_models(data, extension)
    if supervised_predictions["status"] == "available":
        ml_findings.extend(
            f"{item['algorithm'].replace('_', ' ')}: {item['prediction']} "
            f"({item['score_type'].replace('_', ' ')} {item['score']:.3f}); requires human review."
            for item in supervised_predictions["predictions"]
        )
    cnn_prediction = predict_document_image(data, extension)
    if cnn_prediction["status"] == "available":
        ml_findings.append(
            "CNN image analysis: "
            f"{cnn_prediction['prediction']} "
            f"(tampered-pattern probability {cnn_prediction['tampered_probability']:.0%}); "
            "requires human review."
        )
    risk_points = 0
    risk_points += 60 if duplicate_count else 0
    risk_points += min(20, len(quality_findings) * 10)
    if ocr_findings:
        risk_points += 10
    if candidate_name and _normalize(candidate_name) != _normalize(student.full_name):
        risk_points += 30
    if registration and student.registration_number and _normalize(registration) != _normalize(student.registration_number):
        risk_points += 30
    if fields.get("validation", {}).get("is_marks_consistent") is False:
        risk_points += 25
    if any("differs from a prior" in item for item in rule_findings):
        risk_points += 20
    if ml_prediction["status"] == "outlier":
        risk_points += 20
    if cnn_prediction.get("status") == "available" and cnn_prediction["tampered_probability"] >= 0.75:
        risk_points += 20
    if sum(item["prediction"] == "tampered_indicator" for item in supervised_predictions["predictions"]) >= 2:
        risk_points += 15
    score = min(100, risk_points)
    risk_level = "high" if score >= 60 else "medium" if score >= 25 else "low"
    all_findings = [*ocr_findings, *quality_findings, *rule_findings, *ml_findings]
    if not all_findings:
        all_findings.append("No configured risk indicators were detected; this is not proof of authenticity.")
    document.ocr_findings = ocr_findings
    document.rule_findings = [*quality_findings, *rule_findings]
    document.ml_findings = ml_findings
    document.processing_notes = all_findings
    document.status = "pending"
    return {
        "risk_score": score,
        "risk_level": risk_level,
        "indicators": all_findings,
        "ocr_findings": ocr_findings,
        "rule_findings": [*quality_findings, *rule_findings],
        "ml_findings": ml_findings,
        "ml_prediction": ml_prediction,
        "supervised_predictions": supervised_predictions,
        "cnn_prediction": cnn_prediction,
        "model_notes": (
            f"Text extraction method: {extraction_method}. OCR confidence: "
            f"{confidence if confidence is not None else 'not applicable'}. "
            f"Isolation Forest status: {ml_prediction['status']}; "
            f"supervised model status: {supervised_predictions['status']}; "
            f"CNN status: {cnn_prediction['status']}; no result is proof of fraud."
        ),
    }
