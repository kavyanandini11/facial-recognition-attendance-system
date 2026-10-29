"""Accuracy evaluation: hold out a share of each person's samples, train on the
rest, and measure recognition accuracy (plus unknown-rejection rate)."""
from __future__ import annotations

import random

import config
from attendance.recognizer import FaceRecognizer, load_dataset


def evaluate(test_ratio: float = 0.2, seed: int = 42,
             threshold: float = config.CONFIDENCE_THRESHOLD) -> dict:
    data = load_dataset()
    if len(data) < 2:
        raise ValueError("Need at least 2 enrolled users to evaluate.")
    rng = random.Random(seed)
    train, test = {}, []
    for code, imgs in data.items():
        imgs = imgs[:]
        rng.shuffle(imgs)
        n_test = max(1, int(len(imgs) * test_ratio))
        test += [(code, im) for im in imgs[:n_test]]
        train[code] = imgs[n_test:]

    rec = FaceRecognizer()
    rec.train(train)

    correct = wrong = rejected = 0
    per_user: dict[str, list[int]] = {c: [0, 0] for c in data}
    for true_code, img in test:
        pred, _ = rec.predict(img, threshold)
        per_user[true_code][1] += 1
        if pred is None:
            rejected += 1
        elif pred == true_code:
            correct += 1
            per_user[true_code][0] += 1
        else:
            wrong += 1
    total = len(test)
    return {
        "users": len(data),
        "test_samples": total,
        "accuracy": correct / total,
        "misidentified": wrong / total,
        "rejected_as_unknown": rejected / total,
        "threshold": threshold,
        "per_user": {c: (ok / n if n else 0.0) for c, (ok, n) in per_user.items()},
    }
