"""Face detection and preprocessing shared by enrolment, training and recognition.

Keeping one preprocessing path everywhere is what makes LBPH reliable: the
recogniser only ever sees faces cropped, aligned in size and lighting-
normalised exactly the same way.
"""
from __future__ import annotations

import cv2
import numpy as np

import config

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_detector = cv2.CascadeClassifier(_CASCADE_PATH)
# CLAHE evens out uneven lighting much better than plain histogram equalisation.
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return face boxes (x, y, w, h), largest first."""
    faces = _detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=config.MIN_FACE_SIZE
    )
    return sorted((tuple(map(int, f)) for f in faces), key=lambda b: b[2] * b[3], reverse=True)


def preprocess_face(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop with a small margin, resize to FACE_SIZE, normalise lighting, denoise."""
    x, y, w, h = box
    m = int(0.08 * w)
    H, W = gray.shape[:2]
    crop = gray[max(0, y - m): min(H, y + h + m), max(0, x - m): min(W, x + w + m)]
    face = cv2.resize(crop, config.FACE_SIZE, interpolation=cv2.INTER_AREA)
    face = _clahe.apply(face)
    return cv2.GaussianBlur(face, (3, 3), 0)


def augment(face: np.ndarray) -> list[np.ndarray]:
    """Cheap augmentations (flip, brightness) to make LBPH robust to lighting."""
    out = [face, cv2.flip(face, 1)]
    for alpha, beta in ((1.0, 25), (1.0, -25), (1.2, 0)):
        out.append(cv2.convertScaleAbs(face, alpha=alpha, beta=beta))
    return out
