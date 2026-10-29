"""Central configuration.

Every value can be overridden with an environment variable or a `.env` file
in the project root (see `.env.example`).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"          # data/dataset/<user_id>/0001.png
MODEL_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
MODEL_PATH = MODEL_DIR / "lbph_model.yml"
LABELS_PATH = MODEL_DIR / "labels.json"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

# ---------------------------------------------------------------- database
DB_BACKEND = _env("DB_BACKEND", "mysql").lower()   # "mysql" or "sqlite"
MYSQL_CONFIG = {
    "host": _env("MYSQL_HOST", "localhost"),
    "port": int(_env("MYSQL_PORT", "3306")),
    "user": _env("MYSQL_USER", "root"),
    "password": _env("MYSQL_PASSWORD", ""),
    "database": _env("MYSQL_DATABASE", "attendance_db"),
}
SQLITE_PATH = Path(_env("SQLITE_PATH", str(DATA_DIR / "attendance.sqlite3")))

# ---------------------------------------------------------------- camera
CAMERA_INDEX = int(_env("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(_env("FRAME_WIDTH", "960"))
FRAME_HEIGHT = int(_env("FRAME_HEIGHT", "540"))

# ---------------------------------------------------------------- enrolment
FACE_SIZE = (200, 200)                       # (width, height) fed to LBPH
SAMPLES_PER_USER = int(_env("SAMPLES_PER_USER", "60"))
CAPTURE_EVERY_N_FRAMES = int(_env("CAPTURE_EVERY_N_FRAMES", "3"))
MIN_FACE_SIZE = (80, 80)

# ---------------------------------------------------------------- recognition
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
# LBPH returns a distance: LOWER means a closer match.
CONFIDENCE_THRESHOLD = float(_env("CONFIDENCE_THRESHOLD", "65"))
# A face must be recognised in this many consecutive frames before it is logged.
STABLE_FRAMES = int(_env("STABLE_FRAMES", "5"))
# Check-ins after this time (HH:MM) are marked Late ("L") instead of Present ("P").
LATE_AFTER = _env("LATE_AFTER", "09:15")
