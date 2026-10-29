"""Enrolment pipeline: capture faces from the webcam (or import from photos),
preprocess them, and save them to data/dataset/<employee_code>/."""
from __future__ import annotations

from pathlib import Path

import cv2

import config
from attendance.faces import detect_faces, preprocess_face, to_gray
from database.db import Database


def _person_dir(code: str) -> Path:
    d = config.DATASET_DIR / code
    d.mkdir(parents=True, exist_ok=True)
    return d


def _next_index(d: Path) -> int:
    existing = sorted(d.glob("*.png"))
    return int(existing[-1].stem) + 1 if existing else 1


def enrol_from_camera(db: Database, code: str, name: str, department: str | None = None,
                      samples: int = config.SAMPLES_PER_USER) -> int:
    db.add_user(code, name, department)
    out = _person_dir(code)
    idx = _next_index(out)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {config.CAMERA_INDEX}")

    saved, frame_no = 0, 0
    print(f"Enrolling {name} ({code}). Look at the camera; move your head slightly. Press Q to stop.")
    try:
        while saved < samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            gray = to_gray(frame)
            faces = detect_faces(gray)
            if len(faces) == 1:                      # only capture when exactly one face is visible
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                if frame_no % config.CAPTURE_EVERY_N_FRAMES == 0:
                    cv2.imwrite(str(out / f"{idx:04d}.png"), preprocess_face(gray, faces[0]))
                    idx += 1
                    saved += 1
            elif len(faces) > 1:
                cv2.putText(frame, "Only one person in frame, please", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            bar = int(frame.shape[1] * saved / samples)
            cv2.rectangle(frame, (0, frame.shape[0] - 8), (bar, frame.shape[0]), (0, 200, 0), -1)
            cv2.putText(frame, f"{name}: {saved}/{samples}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("Enrolment", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} face samples to {out}")
    return saved


def enrol_from_images(db: Database, code: str, name: str, image_dir: Path,
                      department: str | None = None) -> int:
    """Import existing photos (one person per folder)."""
    db.add_user(code, name, department)
    out = _person_dir(code)
    idx = _next_index(out)
    saved = 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for f in sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts):
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = to_gray(img)
        faces = detect_faces(gray)
        if not faces:
            print(f"  skip {f.name}: no face found")
            continue
        cv2.imwrite(str(out / f"{idx:04d}.png"), preprocess_face(gray, faces[0]))
        idx += 1
        saved += 1
    print(f"Imported {saved} faces for {name} ({code})")
    return saved
