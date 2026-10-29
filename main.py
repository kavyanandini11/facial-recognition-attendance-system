"""Facial Recognition Attendance System — command-line entry point.

    python main.py init-db
    python main.py enrol --code id_014 --name "Asha Rao" --dept CSE
    python main.py enrol --code id_015 --name "Ravi K" --from-dir photos/ravi
    python main.py train
    python main.py run
    python main.py evaluate
    python main.py report --date 2026-10-26
    python main.py users
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import config
from database.db import Database


def cmd_init_db(db: Database, _args) -> None:
    db.init_schema()
    where = config.MYSQL_CONFIG["database"] if db.backend == "mysql" else config.SQLITE_PATH
    print(f"Database ready ({db.backend}: {where})")


def cmd_enrol(db: Database, args) -> None:
    from attendance.enrol import enrol_from_camera, enrol_from_images

    if args.from_dir:
        enrol_from_images(db, args.code, args.name, args.from_dir, args.dept)
    else:
        enrol_from_camera(db, args.code, args.name, args.dept, args.samples)
    print("Now retrain the model: python main.py train")


def cmd_train(db: Database, _args) -> None:
    from attendance.recognizer import FaceRecognizer, load_dataset

    known = {u["employee_code"] for u in db.list_users()}
    data = load_dataset(only=known or None)
    rec = FaceRecognizer()
    n = rec.train(data)
    rec.save()
    print(f"Trained LBPH on {len(data)} users / {n} images (with augmentation) -> {config.MODEL_PATH}")


def cmd_run(db: Database, _args) -> None:
    from attendance.live import run_live

    run_live(db)


def cmd_evaluate(_db: Database, args) -> None:
    from attendance.evaluate import evaluate

    r = evaluate(test_ratio=args.test_ratio, threshold=args.threshold)
    print(f"Users enrolled       : {r['users']}")
    print(f"Held-out samples     : {r['test_samples']}")
    print(f"Recognition accuracy : {r['accuracy']:.1%}")
    print(f"Misidentified        : {r['misidentified']:.1%}")
    print(f"Rejected as unknown  : {r['rejected_as_unknown']:.1%}  (threshold {r['threshold']})")
    worst = sorted(r["per_user"].items(), key=lambda kv: kv[1])[:5]
    print("Lowest per-user accuracy: " + ", ".join(f"{c} {a:.0%}" for c, a in worst))


def cmd_report(db: Database, args) -> None:
    from attendance.report import daily_report, range_report

    if args.start and args.end:
        path = range_report(db, date.fromisoformat(args.start), date.fromisoformat(args.end))
    else:
        day = date.fromisoformat(args.date) if args.date else date.today()
        for r in db.attendance_for(day):
            print(f"{r['employee_code']:<10} {r['name']:<24} {r['check_in']}  {r['status']}")
        path = daily_report(db, day)
    print(f"Report written to {path}")


def cmd_users(db: Database, args) -> None:
    if args.delete:
        ok = db.delete_user(args.delete)
        print("Deleted." if ok else "No such user.")
        return
    users = db.list_users()
    for u in users:
        print(f"{u['employee_code']:<10} {u['name']:<24} {u['department'] or ''}")
    print(f"{len(users)} enrolled")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Facial Recognition Attendance System (LBPH + MySQL)")
    p.add_argument("--db", choices=["mysql", "sqlite"], help="override DB_BACKEND")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create database and tables")

    e = sub.add_parser("enrol", help="enrol a user from the webcam or a photo folder")
    e.add_argument("--code", required=True, help="employee/student code, e.g. id_014")
    e.add_argument("--name", required=True)
    e.add_argument("--dept")
    e.add_argument("--samples", type=int, default=config.SAMPLES_PER_USER)
    e.add_argument("--from-dir", help="import photos from this folder instead of the camera")

    sub.add_parser("train", help="train the LBPH recogniser on all enrolled faces")
    sub.add_parser("run", help="start live attendance capture")

    v = sub.add_parser("evaluate", help="measure recognition accuracy on held-out samples")
    v.add_argument("--test-ratio", type=float, default=0.2)
    v.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)

    r = sub.add_parser("report", help="print and export attendance to CSV")
    r.add_argument("--date", help="YYYY-MM-DD (default today)")
    r.add_argument("--start")
    r.add_argument("--end")

    u = sub.add_parser("users", help="list or delete enrolled users")
    u.add_argument("--delete", metavar="CODE")
    return p


COMMANDS = {"init-db": cmd_init_db, "enrol": cmd_enrol, "train": cmd_train, "run": cmd_run,
            "evaluate": cmd_evaluate, "report": cmd_report, "users": cmd_users}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    db = Database(args.db)
    try:
        COMMANDS[args.command](db, args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
