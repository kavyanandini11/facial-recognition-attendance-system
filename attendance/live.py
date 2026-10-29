"""Live recognition loop: camera → detect → LBPH match → stable-frame vote → DB."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

import cv2

import config
from attendance.faces import detect_faces, preprocess_face, to_gray
from attendance.recognizer import FaceRecognizer
from database.db import Database

GREEN, AMBER, RED, WHITE = (60, 200, 60), (0, 180, 255), (60, 60, 230), (255, 255, 255)


class AttendanceSession:
    """Holds per-session state so the logic is testable without a camera."""

    def __init__(self, db: Database, recognizer: FaceRecognizer):
        self.db = db
        self.rec = recognizer
        self.users = {u["employee_code"]: u for u in db.list_users()}
        self.streak: dict[str, int] = defaultdict(int)
        self.logged_today: set[str] = set()
        self.feed: deque[str] = deque(maxlen=6)     # last check-ins shown on screen

    def process_face(self, face) -> tuple[str | None, float, bool]:
        """Return (code, distance, just_logged)."""
        code, dist = self.rec.predict(face)
        if code is None:
            return None, dist, False
        self.streak[code] += 1
        if code in self.logged_today or self.streak[code] < config.STABLE_FRAMES:
            return code, dist, False
        user = self.users.get(code) or self.db.get_user_by_code(code)
        if not user:
            return code, dist, False
        created, status = self.db.mark_attendance(user["id"], dist)
        self.logged_today.add(code)
        if created:
            stamp = datetime.now().strftime("%H:%M")
            self.feed.appendleft(f"{code} · {stamp} · {status}")
            print(f"[{stamp}] {code} {user['name']} -> {status} (distance {dist:.1f})")
        return code, dist, created

    def reset_unseen(self, seen: set[str]) -> None:
        for code in list(self.streak):
            if code not in seen:
                self.streak[code] = 0


def _draw_feed(frame, feed) -> None:
    x0 = frame.shape[1] - 260
    cv2.rectangle(frame, (x0 - 10, 10), (frame.shape[1] - 10, 40 + 26 * max(1, len(feed))), (30, 30, 30), -1)
    cv2.putText(frame, "attendance", (x0, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)
    for i, line in enumerate(feed):
        cv2.putText(frame, line, (x0, 58 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)


def run_live(db: Database) -> None:
    rec = FaceRecognizer().load()
    session = AttendanceSession(db, rec)
    # Pre-load anyone already marked today so they aren't re-announced.
    for row in db.attendance_for(datetime.now().date()):
        session.logged_today.add(row["employee_code"])

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {config.CAMERA_INDEX}")
    print("Live attendance running. Press Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = to_gray(frame)
            seen = set()
            for box in detect_faces(gray):
                code, dist, _ = session.process_face(preprocess_face(gray, box))
                x, y, w, h = box
                if code is None:
                    colour, label = RED, f"Unknown ({dist:.0f})"
                else:
                    seen.add(code)
                    done = code in session.logged_today
                    colour = GREEN if done else AMBER
                    name = session.users.get(code, {}).get("name", code)
                    label = f"{name} {'OK' if done else '...'} ({dist:.0f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                cv2.rectangle(frame, (x, y - 28), (x + w, y), colour, -1)
                cv2.putText(frame, label, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            session.reset_unseen(seen)
            _draw_feed(frame, session.feed)
            cv2.putText(frame, "LBPH MATCH  |  MySQL auto-logged" if db.backend == "mysql"
                        else "LBPH MATCH  |  SQLite auto-logged",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
            cv2.imshow("Facial Recognition Attendance", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
