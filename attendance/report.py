"""CSV attendance reports."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import config
from database.db import Database


def daily_report(db: Database, day: date) -> Path:
    present = {r["employee_code"]: r for r in db.attendance_for(day)}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / f"attendance_{day.isoformat()}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["employee_code", "name", "department", "date", "check_in", "status"])
        for u in db.list_users():
            r = present.get(u["employee_code"])
            w.writerow([u["employee_code"], u["name"], u["department"] or "", day.isoformat(),
                        r["check_in"] if r else "", r["status"] if r else "A"])
    return path


def range_report(db: Database, start: date, end: date) -> Path:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / f"attendance_{start.isoformat()}_to_{end.isoformat()}.csv"
    rows = db.attendance_range(start, end)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "employee_code", "name", "check_in", "status", "confidence"])
        w.writeheader()
        w.writerows(rows)
    return path
