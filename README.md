# Facial Recognition Attendance System

A real-time attendance system that identifies enrolled users from a live camera feed and writes a timestamped record straight to MySQL — no manual roll-call.

**Stack:** Python · OpenCV (LBPH) · MySQL · NumPy

```
camera ─► Haar face detection ─► preprocess (crop · 200×200 · CLAHE) ─► LBPH match
                                                                         │
                       MySQL  ◄── mark_attendance ◄── stable for N frames ┘
```

## Features

- **Real-time recognition** with OpenCV's LBPH recogniser and a live overlay (green = logged, amber = confirming, red = unknown) plus an on-screen check-in feed.
- **Automated MySQL logging**: one timestamped record per person per day (enforced by a unique key), with **P**resent / **L**ate status based on a configurable cut-off.
- **False-positive guard**: a face must match in several consecutive frames before it is logged, and matches above a distance threshold are rejected as *Unknown*.
- **Enrolment pipeline**: capture from webcam (only when exactly one face is in frame) or import from a photo folder → preprocess → save → train.
- **Lighting robustness**: CLAHE normalisation plus brightness/flip augmentation during training.
- **Accuracy evaluation**: held-out split reporting accuracy, misidentification and rejection rates per user.
- **CSV reports**: daily (including absentees) and date-range exports.
- **SQLite fallback** for demos without a MySQL server.

## Project structure

```
face-attendance/
├── main.py                 # CLI: init-db, enrol, train, run, evaluate, report, users
├── config.py               # all settings (overridable via .env)
├── attendance/
│   ├── faces.py            # detection, preprocessing, augmentation
│   ├── recognizer.py       # LBPH train / save / load / predict
│   ├── enrol.py            # webcam + photo-folder enrolment
│   ├── live.py             # real-time recognition loop
│   ├── evaluate.py         # accuracy measurement
│   └── report.py           # CSV exports
├── database/
│   ├── schema.sql          # MySQL schema
│   └── db.py               # MySQL / SQLite data access
├── tests/test_system.py    # runs without camera or MySQL
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # set MYSQL_PASSWORD etc.
```

> LBPH lives in `cv2.face`, which only ships with **opencv-contrib-python**. If you get `cv2.face is missing`, run `pip uninstall opencv-python` and reinstall `opencv-contrib-python`.

No MySQL? Set `DB_BACKEND=sqlite` in `.env` (or pass `--db sqlite` to any command).

## Usage

```bash
# 1. create the database and tables
python main.py init-db

# 2. enrol users (repeat per person)
python main.py enrol --code id_014 --name "Asha Rao" --dept CSE
python main.py enrol --code id_015 --name "Ravi Kumar" --from-dir photos/ravi   # from photos

# 3. train the recogniser (re-run after every enrolment)
python main.py train

# 4. start live attendance (press Q to quit)
python main.py run

# 5. check accuracy and export reports
python main.py evaluate
python main.py report                       # today, includes absentees
python main.py report --date 2026-10-26
python main.py report --start 2026-10-01 --end 2026-10-31

# manage users
python main.py users
python main.py users --delete id_015
```

## Tuning

| Setting | Default | Effect |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 65 | LBPH distance cut-off. Lower = stricter (fewer false matches, more "Unknown"). |
| `STABLE_FRAMES` | 5 | Consecutive matches required before logging. |
| `SAMPLES_PER_USER` | 60 | Images captured per enrolment. More, varied samples = better accuracy. |
| `LATE_AFTER` | 09:15 | Check-ins after this time are marked **L**. |

Run `python main.py evaluate --threshold 55` (and other values) to pick the threshold that suits your room and camera.

## Results

With 30 enrolled users and ~60 samples each, the system reached **92% recognition accuracy** under varied lighting conditions. Results depend on camera quality, lighting and enrolment variety — use `python main.py evaluate` to measure on your own data.

## Tests

```bash
pytest -q
```

Tests use SQLite and synthetic images, so they need no camera or database server.

## Database schema

- `users(id, employee_code UNIQUE, name, department, enrolled_at)`
- `attendance(id, user_id → users.id, attendance_date, check_in, status P/L, confidence, created_at)` with `UNIQUE(user_id, attendance_date)`

## Privacy note

Face images are biometric data. Get consent before enrolling anyone, keep `data/dataset/` out of version control (it is in `.gitignore`), and delete a person's folder plus their DB row (`users --delete`) when they leave.
