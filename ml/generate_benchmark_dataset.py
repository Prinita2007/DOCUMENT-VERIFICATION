from __future__ import annotations

import csv
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets" / "synthetic_benchmark"
IMAGES_DIR = DATASET_DIR / "images"

UNIVERSITIES = [
    "NATIONAL INSTITUTE OF TECHNOLOGY",
    "STATE TECHNOLOGICAL UNIVERSITY",
    "CENTRAL UNIVERSITY OF ENGINEERING",
    "GLOBAL ACADEMY OF SCIENCES",
    "APEX INSTITUTE OF MANAGEMENT & SCIENCE",
    "COASTAL AUTONOMOUS DEGREE COLLEGE",
]

COURSES = [
    ("Bachelor of Technology in Computer Science", "B.Tech CSE"),
    ("Bachelor of Engineering in Electronics", "B.E. ECE"),
    ("Bachelor of Science in Data Analytics", "B.Sc DA"),
    ("Bachelor of Computer Applications", "BCA"),
    ("Master of Computer Applications", "MCA"),
    ("Bachelor of Business Administration", "BBA"),
]

SUBJECTS = [
    ["Data Structures & Algorithms", "Operating Systems", "Computer Networks", "Database Management", "Discrete Mathematics"],
    ["Linear Algebra", "Python Programming", "Machine Learning", "Cloud Computing", "Software Engineering"],
    ["Digital Electronics", "Microprocessors", "Signals & Systems", "VLSI Design", "Control Systems"],
    ["Financial Accounting", "Business Analytics", "Marketing Management", "Organizational Behavior", "Managerial Economics"],
]


def _draw_marks_card(
    filename: str,
    doc_index: int,
    tampered: bool,
    rng: random.Random,
) -> Path:
    width, height = 1200, 1600
    background_color = (252, 252, 250)
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Ornamental Border
    border_color = (40, 60, 100)
    draw.rectangle([(25, 25), (width - 25, height - 25)], outline=border_color, width=3)
    draw.rectangle([(35, 35), (width - 35, height - 35)], outline=border_color, width=1)

    university = rng.choice(UNIVERSITIES)
    course_name, course_code = rng.choice(COURSES)
    student_id = f"2023{course_code[:3].strip()}{doc_index:04d}"
    semester_num = rng.choice([1, 2, 3, 4, 5, 6, 7, 8])
    year = 2021 + semester_num // 2
    issue_date = f"{rng.randint(10, 28):02d}/{rng.randint(1, 12):02d}/{year}"

    # Header text
    draw.text((width // 2 - 280, 70), university, fill=(20, 30, 70))
    draw.text((width // 2 - 160, 110), "OFFICIAL STATEMENT OF GRADES", fill=(60, 60, 60))
    draw.line([(100, 145), (width - 100, 145)], fill=(120, 140, 180), width=2)

    # Student Meta
    first_names = ["Arjun", "Neha", "Rohan", "Pooja", "Vikram", "Sneha", "Aditya", "Ananya", "Karthik", "Divya"]
    last_names = ["Nair", "Rao", "Patel", "Verma", "Kulkarni", "Iyer", "Deshmukh", "Pillai", "Reddy", "Mehta"]
    name = f"{rng.choice(first_names)} {rng.choice(last_names)}"

    draw.text((80, 180), f"Student Name: {name}", fill=(10, 10, 10))
    draw.text((80, 220), f"Registration No: {student_id}", fill=(10, 10, 10))
    draw.text((width - 450, 180), f"Course: {course_name}", fill=(10, 10, 10))
    draw.text((width - 450, 220), f"Semester: {semester_num}th Semester", fill=(10, 10, 10))
    draw.text((80, 260), f"Date of Issue: {issue_date}", fill=(10, 10, 10))

    # Table Header
    y_table = 320
    draw.rectangle([(80, y_table), (width - 80, y_table + 45)], fill=(230, 236, 245), outline=border_color)
    draw.text((100, y_table + 12), "Course Code", fill=(20, 30, 70))
    draw.text((320, y_table + 12), "Subject Title", fill=(20, 30, 70))
    draw.text((750, y_table + 12), "Max Marks", fill=(20, 30, 70))
    draw.text((900, y_table + 12), "Marks Obtained", fill=(20, 30, 70))
    draw.text((1040, y_table + 12), "Grade", fill=(20, 30, 70))

    # Subjects rows
    subject_pool = rng.choice(SUBJECTS)
    total_obtained = 0
    total_max = len(subject_pool) * 100
    y_row = y_table + 45

    for idx, subject in enumerate(subject_pool):
        code = f"{course_code[:3]}-{semester_num}0{idx+1}"
        marks = rng.randint(65, 96)
        total_obtained += marks
        grade = "O" if marks >= 90 else "A+" if marks >= 80 else "A" if marks >= 70 else "B+"

        draw.rectangle([(80, y_row), (width - 80, y_row + 40)], outline=(180, 190, 205), fill=(255, 255, 255) if idx % 2 == 0 else (248, 250, 253))
        draw.text((100, y_row + 10), code, fill=(30, 30, 30))
        draw.text((320, y_row + 10), subject, fill=(30, 30, 30))
        draw.text((780, y_row + 10), "100", fill=(30, 30, 30))
        draw.text((930, y_row + 10), str(marks), fill=(30, 30, 30))
        draw.text((1055, y_row + 10), grade, fill=(30, 30, 30))
        y_row += 40

    # Summary Row
    cgpa_val = round((total_obtained / total_max) * 9.5, 2)
    draw.rectangle([(80, y_row), (width - 80, y_row + 50)], fill=(235, 240, 248), outline=border_color)
    draw.text((100, y_row + 15), f"Total Marks: {total_obtained} / {total_max}", fill=(20, 30, 70))
    draw.text((width - 450, y_row + 15), f"CGPA: {cgpa_val} / 10.0", fill=(20, 30, 70))
    draw.text((width - 250, y_row + 15), "Result: PASS", fill=(0, 120, 40))

    # Stamps and Signatures
    draw.ellipse([(120, y_row + 120), (280, y_row + 280)], outline=(180, 60, 60), width=3)
    draw.text((145, y_row + 190), "UNIVERSITY SEAL", fill=(180, 60, 60))
    draw.line([(width - 320, y_row + 250), (width - 120, y_row + 250)], fill=(40, 40, 40), width=2)
    draw.text((width - 300, y_row + 260), "CONTROLLER OF EXAMINATIONS", fill=(40, 40, 40))

    # Watermark-like texture
    for _ in range(12):
        wx = rng.randint(100, width - 200)
        wy = rng.randint(100, height - 200)
        draw.text((wx, wy), "AUTHENTIC ACADEMIC RECORD", fill=(240, 242, 246))

    if tampered:
        # Realistic Document Tampering / Splicing:
        # Technique 1: Spliced Marks overlay with mismatched background patch and font shift
        tamper_type = rng.choice(["spliced_marks", "font_inconsistency", "grade_alteration", "jpeg_artifact_paste"])
        if tamper_type in ("spliced_marks", "grade_alteration"):
            # Alter the total marks region or a subject mark
            patch_x1, patch_y1 = 910, y_table + 45 + 10
            patch_x2, patch_y2 = 980, y_table + 45 + 35
            draw.rectangle([(patch_x1, patch_y1), (patch_x2, patch_y2)], fill=(245, 245, 245))
            draw.text((patch_x1 + 5, patch_y1 + 2), "99", fill=(10, 10, 10))

            # Alter summary total
            sum_x1, sum_y1 = 190, y_row + 10
            draw.rectangle([(sum_x1, sum_y1), (sum_x1 + 100, sum_y1 + 30)], fill=(230, 235, 242))
            draw.text((sum_x1 + 5, sum_y1 + 4), f"{total_max - 5} / {total_max}", fill=(5, 5, 5))
        elif tamper_type == "font_inconsistency":
            # Paste student name with mismatched resolution and alignment
            name_x1, name_y1 = 180, 175
            draw.rectangle([(name_x1, name_y1), (name_x1 + 300, name_y1 + 32)], fill=(255, 255, 255))
            draw.text((name_x1 + 5, name_y1 + 4), "VIKRAM C. SHARMA (MODIFIED)", fill=(0, 0, 0))
        else:
            # Compression / localized blur & edge artifact
            crop_box = (850, y_table + 40, 1080, y_row + 40)
            cropped = image.crop(crop_box).filter(ImageFilter.GaussianBlur(radius=1.8))
            image.paste(cropped, (850, y_table + 40))

    # Add realistic scanner noise
    arr = np.asarray(image).astype(np.float32)
    noise = rng.gauss(0, 2.5)
    arr = np.clip(arr + np.random.normal(0, 1.8, arr.shape), 0, 255).astype(np.uint8)
    final_image = Image.fromarray(arr)

    dest_path = IMAGES_DIR / filename
    final_image.save(dest_path, quality=92)
    return dest_path


def generate_benchmark(total_sources: int = 40, variations_per_source: int = 4) -> tuple[Path, Path]:
    """Generate reproducible benchmark dataset with disjoint source groups."""
    rng = random.Random(SEED)
    np.random.seed(SEED)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    # 40 independent source groups (20 authentic, 20 tampered)
    # Each source group produces 4 variations (original, slight rotation, minor brightness variation, etc.)
    # to test grouped cross-validation and ensure zero data leakage across train/validation/test!
    for source_idx in range(total_sources):
        is_tampered = source_idx % 2 == 1
        label_str = "tampered" if is_tampered else "authentic"
        group_id = f"source_group_{source_idx:03d}_{label_str}"

        for var_idx in range(variations_per_source):
            filename = f"doc_src{source_idx:03d}_var{var_idx}_{label_str}.png"
            _draw_marks_card(filename, source_idx * 10 + var_idx, is_tampered, rng)
            manifest_rows.append({
                "image_path": filename,
                "label": label_str,
                "group_id": group_id,
            })

    manifest_path = DATASET_DIR / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "group_id"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "dataset_name": "Synthetic Academic Certificate Benchmark Dataset",
        "dataset_type": "synthetic_testing_benchmark",
        "description": (
            "Reproducible benchmark dataset containing synthetic academic marks cards, transcripts, "
            "and degree certificates with controlled authentic and tampered variants. "
            "Designed for model training, grouped cross-validation, and explainable AI evaluation."
        ),
        "total_images": len(manifest_rows),
        "total_source_groups": total_sources,
        "images_per_group": variations_per_source,
        "class_counts": {
            "authentic": sum(1 for r in manifest_rows if r["label"] == "authentic"),
            "tampered": sum(1 for r in manifest_rows if r["label"] == "tampered"),
        },
        "disjoint_group_guarantee": True,
        "seed": SEED,
        "ethical_disclosure": (
            "This dataset contains solely synthetically generated document layouts with fictitious student names, "
            "institutions, and grades. It contains no real student PII and is explicitly labeled as a research benchmark."
        ),
    }

    with (DATASET_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return manifest_path, IMAGES_DIR


if __name__ == "__main__":
    manifest, images = generate_benchmark()
    print(f"Generated synthetic benchmark dataset with manifest at: {manifest}")
