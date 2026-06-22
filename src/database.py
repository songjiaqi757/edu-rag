import sqlite3
from datetime import datetime

from src.config import DB_PATH


def _ensure_column(conn: sqlite3.Connection, name: str, definition: str) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(qa_logs)").fetchall()}
    if name not in columns:
        conn.execute(f"ALTER TABLE qa_logs ADD COLUMN {name} {definition}")


def init_db() -> None:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    is_answered INTEGER NOT NULL DEFAULT 1,
                    top_score REAL NOT NULL DEFAULT 0,
                    course_sources TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            _ensure_column(conn, "is_answered", "INTEGER NOT NULL DEFAULT 1")
            _ensure_column(conn, "top_score", "REAL NOT NULL DEFAULT 0")
            _ensure_column(conn, "course_sources", "TEXT NOT NULL DEFAULT ''")
            conn.commit()
    except sqlite3.Error:
        pass


def save_qa_log(
    question: str,
    answer: str,
    sources: str,
    is_answered: bool,
    top_score: float,
    course_sources: str,
) -> bool:
    try:
        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO qa_logs (
                    question, answer, sources, is_answered, top_score,
                    course_sources, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question,
                    answer,
                    sources,
                    int(is_answered),
                    top_score,
                    course_sources,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            conn.commit()
        return True
    except sqlite3.Error:
        return False


def get_qa_logs() -> list[dict]:
    try:
        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    id, question, answer, sources, is_answered, top_score,
                    course_sources, created_at
                FROM qa_logs
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []
