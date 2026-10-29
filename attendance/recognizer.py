"""LBPH recogniser wrapper: training, saving/loading, prediction."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import config
from attendance.faces import augment


def _new_model():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face is missing. Install opencv-contrib-python "
            "(and uninstall plain opencv-python): pip install opencv-contrib-python"
        )
    return cv2.face.LBPHFaceRecognizer_create(
        radius=config.LBPH_RADIUS, neighbors=config.LBPH_NEIGHBORS,
        grid_x=config.LBPH_GRID_X, grid_y=config.LBPH_GRID_Y,
    )


def load_dataset(dataset_dir: Path = config.DATASET_DIR,
                 only: set[str] | None = None) -> dict[str, list[np.ndarray]]:
    """Return {employee_code: [face images]} from data/dataset/<code>/*.png."""
    data: dict[str, list[np.ndarray]] = {}
    if not dataset_dir.exists():
        return data
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        if only and person_dir.name not in only:
            continue
        imgs = []
        for f in sorted(person_dir.glob("*.png")):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape[::-1] != config.FACE_SIZE:
                    img = cv2.resize(img, config.FACE_SIZE)
                imgs.append(img)
        if imgs:
            data[person_dir.name] = imgs
    return data


class FaceRecognizer:
    def __init__(self):
        self.model = None
        self.labels: dict[int, str] = {}   # numeric label -> employee_code

    # ---------------------------------------------------------------- train
    def train(self, data: dict[str, list[np.ndarray]], use_augmentation: bool = True) -> int:
        if len(data) == 0:
            raise ValueError("No training data. Enrol at least one user first.")
        faces, ids = [], []
        self.labels = {}
        for label, (code, imgs) in enumerate(sorted(data.items())):
            self.labels[label] = code
            for img in imgs:
                for f in (augment(img) if use_augmentation else [img]):
                    faces.append(f)
                    ids.append(label)
        self.model = _new_model()
        self.model.train(faces, np.array(ids, dtype=np.int32))
        return len(faces)

    def save(self, model_path: Path = config.MODEL_PATH, labels_path: Path = config.LABELS_PATH) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.write(str(model_path))
        labels_path.write_text(json.dumps({str(k): v for k, v in self.labels.items()}, indent=2))

    def load(self, model_path: Path = config.MODEL_PATH, labels_path: Path = config.LABELS_PATH) -> "FaceRecognizer":
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model at {model_path}. Run: python main.py train")
        self.model = _new_model()
        self.model.read(str(model_path))
        self.labels = {int(k): v for k, v in json.loads(labels_path.read_text()).items()}
        return self

    # ---------------------------------------------------------------- predict
    def predict(self, face: np.ndarray, threshold: float = config.CONFIDENCE_THRESHOLD) -> tuple[str | None, float]:
        """Return (employee_code or None if unknown, LBPH distance)."""
        label, distance = self.model.predict(face)
        if distance > threshold:
            return None, float(distance)
        return self.labels.get(int(label)), float(distance)
