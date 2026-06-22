import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "qa_logs.db"


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
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
    except sqlite3.Error:
        pass


def save_qa_log(question: str, answer: str, sources: str) -> bool:
    try:
        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO qa_logs (question, answer, sources, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    question,
                    answer,
                    sources,
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
                SELECT id, question, answer, sources, created_at
                FROM qa_logs
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []
