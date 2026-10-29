"""Database layer.

Supports MySQL (default, as in production) and SQLite (zero-setup fallback for
demos and tests). Switch with DB_BACKEND=sqlite in `.env`.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, time
from typing import Iterator

import config

# SQLite equivalent of database/schema.sql
_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_code  TEXT NOT NULL UNIQUE,
    name           TEXT NOT NULL,
    department     TEXT,
    enrolled_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS attendance (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    attendance_date  TEXT NOT NULL,
    check_in         TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'P',
    confidence       REAL,
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, attendance_date)
);
CREATE INDEX IF NOT EXISTS idx_date ON attendance(attendance_date);
"""


class Database:
    def __init__(self, backend: str | None = None):
        self.backend = (backend or config.DB_BACKEND).lower()
        if self.backend not in {"mysql", "sqlite"}:
            raise ValueError(f"Unknown DB_BACKEND {self.backend!r}")
        self.ph = "%s" if self.backend == "mysql" else "?"   # placeholder style

    # ------------------------------------------------------------ connection
    def _connect(self, with_db: bool = True):
        if self.backend == "sqlite":
            config.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(config.SQLITE_PATH)
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
        import mysql.connector  # imported lazily so SQLite works without it

        cfg = dict(config.MYSQL_CONFIG)
        if not with_db:
            cfg.pop("database")
        return mysql.connector.connect(**cfg)

    @contextmanager
    def cursor(self) -> Iterator:
        conn = self._connect()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def _q(self, sql: str) -> str:
        return sql.replace("%s", self.ph)

    # ------------------------------------------------------------ schema
    def init_schema(self) -> None:
        if self.backend == "sqlite":
            conn = self._connect()
            conn.executescript(_SQLITE_SCHEMA)
            conn.commit()
            conn.close()
            return
        sql = config.SCHEMA_PATH.read_text(encoding="utf-8")
        sql = re.sub(r"--.*", "", sql)
        db_name = config.MYSQL_CONFIG["database"]
        sql = sql.replace("attendance_db", db_name)
        conn = self._connect(with_db=False)
        cur = conn.cursor()
        for stmt in filter(None, (s.strip() for s in sql.split(";"))):
            cur.execute(stmt)
        conn.commit()
        cur.close()
        conn.close()

    # ------------------------------------------------------------ users
    def add_user(self, employee_code: str, name: str, department: str | None = None) -> int:
        existing = self.get_user_by_code(employee_code)
        if existing:
            with self.cursor() as cur:
                cur.execute(self._q("UPDATE users SET name=%s, department=%s WHERE id=%s"),
                            (name, department, existing["id"]))
            return existing["id"]
        with self.cursor() as cur:
            cur.execute(self._q("INSERT INTO users (employee_code, name, department) VALUES (%s, %s, %s)"),
                        (employee_code, name, department))
            return cur.lastrowid

    def get_user_by_code(self, employee_code: str) -> dict | None:
        with self.cursor() as cur:
            cur.execute(self._q("SELECT id, employee_code, name, department FROM users WHERE employee_code=%s"),
                        (employee_code,))
            row = cur.fetchone()
        return _user(row) if row else None

    def get_user(self, user_id: int) -> dict | None:
        with self.cursor() as cur:
            cur.execute(self._q("SELECT id, employee_code, name, department FROM users WHERE id=%s"), (user_id,))
            row = cur.fetchone()
        return _user(row) if row else None

    def list_users(self) -> list[dict]:
        with self.cursor() as cur:
            cur.execute("SELECT id, employee_code, name, department FROM users ORDER BY employee_code")
            return [_user(r) for r in cur.fetchall()]

    def delete_user(self, employee_code: str) -> bool:
        with self.cursor() as cur:
            cur.execute(self._q("DELETE FROM attendance WHERE user_id IN "
                                "(SELECT id FROM users WHERE employee_code=%s)"), (employee_code,))
            cur.execute(self._q("DELETE FROM users WHERE employee_code=%s"), (employee_code,))
            return cur.rowcount > 0

    # ------------------------------------------------------------ attendance
    def mark_attendance(self, user_id: int, confidence: float | None = None,
                        when: datetime | None = None) -> tuple[bool, str]:
        """Insert today's check-in. Returns (created, status). Idempotent per day."""
        when = when or datetime.now()
        day, t = when.date(), when.time().replace(microsecond=0)
        late_h, late_m = map(int, config.LATE_AFTER.split(":"))
        status = "L" if t > time(late_h, late_m) else "P"
        with self.cursor() as cur:
            cur.execute(self._q("SELECT status FROM attendance WHERE user_id=%s AND attendance_date=%s"),
                        (user_id, day.isoformat()))
            row = cur.fetchone()
            if row:
                return False, row[0]
            cur.execute(self._q("INSERT INTO attendance (user_id, attendance_date, check_in, status, confidence) "
                                "VALUES (%s, %s, %s, %s, %s)"),
                        (user_id, day.isoformat(), t.strftime("%H:%M:%S"), status,
                         None if confidence is None else round(float(confidence), 2)))
        return True, status

    def attendance_for(self, day: date) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(self._q(
                "SELECT u.employee_code, u.name, u.department, a.check_in, a.status, a.confidence "
                "FROM attendance a JOIN users u ON u.id = a.user_id "
                "WHERE a.attendance_date=%s ORDER BY a.check_in"), (day.isoformat(),))
            rows = cur.fetchall()
        return [{"employee_code": r[0], "name": r[1], "department": r[2],
                 "check_in": str(r[3])[:8], "status": r[4], "confidence": r[5]} for r in rows]

    def attendance_range(self, start: date, end: date) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(self._q(
                "SELECT a.attendance_date, u.employee_code, u.name, a.check_in, a.status, a.confidence "
                "FROM attendance a JOIN users u ON u.id = a.user_id "
                "WHERE a.attendance_date BETWEEN %s AND %s ORDER BY a.attendance_date, a.check_in"),
                (start.isoformat(), end.isoformat()))
            rows = cur.fetchall()
        return [{"date": str(r[0]), "employee_code": r[1], "name": r[2],
                 "check_in": str(r[3])[:8], "status": r[4], "confidence": r[5]} for r in rows]


def _user(row) -> dict:
    return {"id": row[0], "employee_code": row[1], "name": row[2], "department": row[3]}
