"""Tests run without a camera or MySQL: SQLite + synthetic face-like images."""
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "t.sqlite3")
    monkeypatch.setattr(config, "DATASET_DIR", tmp_path / "dataset")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "m.yml")
    monkeypatch.setattr(config, "LABELS_PATH", tmp_path / "models" / "labels.json")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    from database.db import Database
    db = Database("sqlite")
    db.init_schema()
    return db


def _synthetic_person(seed: int, n: int = 20):
    """A fixed random texture per 'person', with per-sample noise + shifts."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 255, config.FACE_SIZE[::-1], dtype=np.uint8), (9, 9), 0)
    out = []
    for _ in range(n):
        noise = rng.normal(0, 6, base.shape)
        img = np.clip(base + noise + rng.integers(-15, 15), 0, 255).astype(np.uint8)
        M = np.float32([[1, 0, rng.integers(-3, 4)], [0, 1, rng.integers(-3, 4)]])
        out.append(cv2.warpAffine(img, M, config.FACE_SIZE, borderMode=cv2.BORDER_REFLECT))
    return out


def _write_dataset(n_people=5):
    for p in range(n_people):
        d = config.DATASET_DIR / f"id_{p:03d}"
        d.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(_synthetic_person(p)):
            cv2.imwrite(str(d / f"{i + 1:04d}.png"), img)


def test_users_and_attendance_once_per_day(tmp_env):
    db = tmp_env
    uid = db.add_user("id_014", "Asha", "CSE")
    created, status = db.mark_attendance(uid, 40.0, datetime(2026, 10, 26, 9, 2))
    assert created and status == "P"
    created, _ = db.mark_attendance(uid, 38.0, datetime(2026, 10, 26, 10, 0))
    assert not created                                   # no duplicate per day
    _, status = db.mark_attendance(uid, 40.0, datetime(2026, 10, 27, 9, 30))
    assert status == "L"                                 # after LATE_AFTER
    assert db.attendance_for(datetime(2026, 10, 26).date())[0]["check_in"] == "09:02:00"


def test_train_predict_and_evaluate(tmp_env):
    from attendance.evaluate import evaluate
    from attendance.recognizer import FaceRecognizer, load_dataset

    _write_dataset()
    rec = FaceRecognizer()
    rec.train(load_dataset())
    rec.save()
    rec2 = FaceRecognizer().load()
    code, dist = rec2.predict(_synthetic_person(3, 1)[0], threshold=1e9)
    assert code == "id_003"
    r = evaluate(threshold=1e9)
    assert r["accuracy"] >= 0.9


def test_session_requires_stable_frames(tmp_env, monkeypatch):
    from attendance.live import AttendanceSession
    from attendance.recognizer import FaceRecognizer, load_dataset

    _write_dataset(3)
    for p in range(3):
        tmp_env.add_user(f"id_{p:03d}", f"Person {p}")
    rec = FaceRecognizer()
    rec.train(load_dataset())
    monkeypatch.setattr(config, "STABLE_FRAMES", 3)
    monkeypatch.setattr(rec, "predict", lambda face, threshold=None: ("id_001", 30.0))
    s = AttendanceSession(tmp_env, rec)
    face = _synthetic_person(1, 1)[0]
    assert [s.process_face(face)[2] for _ in range(4)] == [False, False, True, False]
    assert len(tmp_env.attendance_for(datetime.now().date())) == 1


def test_daily_report_marks_absent(tmp_env):
    from attendance.report import daily_report

    a = tmp_env.add_user("id_001", "A")
    tmp_env.add_user("id_002", "B")
    tmp_env.mark_attendance(a, 30.0, datetime(2026, 10, 26, 9, 0))
    text = daily_report(tmp_env, datetime(2026, 10, 26).date()).read_text()
    assert "id_001,A,,2026-10-26,09:00:00,P" in text
    assert "id_002,B,,2026-10-26,,A" in text
