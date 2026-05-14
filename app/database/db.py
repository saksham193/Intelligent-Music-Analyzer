"""SQLite storage for recommendation history."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from app.config.settings import DATABASE_PATH


def get_connection(db_path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    """Create a SQLite connection and ensure the parent folder exists."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def create_tables(db_path: str | Path = DATABASE_PATH) -> None:
    """Create all required database tables."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                voice_emotion TEXT NOT NULL,
                text_sentiment TEXT NOT NULL,
                final_emotion TEXT NOT NULL,
                transcript TEXT,
                language TEXT,
                bpm REAL,
                recommended_songs TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def insert_history(
    voice_emotion: str,
    text_sentiment: str,
    final_emotion: str,
    transcript: str,
    language: str,
    bpm: float,
    recommended_songs: List[Dict[str, str]],
    db_path: str | Path = DATABASE_PATH,
) -> None:
    """Insert one analysis session into history."""
    create_tables(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_history (
                voice_emotion,
                text_sentiment,
                final_emotion,
                transcript,
                language,
                bpm,
                recommended_songs
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                voice_emotion,
                text_sentiment,
                final_emotion,
                transcript,
                language,
                bpm,
                json.dumps(recommended_songs),
            ),
        )


def fetch_history(limit: int = 20, db_path: str | Path = DATABASE_PATH) -> List[Dict[str, object]]:
    """Fetch recent user history records."""
    create_tables(db_path)
    with get_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM user_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    history: List[Dict[str, object]] = []
    for row in rows:
        item = dict(row)
        try:
            item["recommended_songs"] = json.loads(item.get("recommended_songs") or "[]")
        except json.JSONDecodeError:
            item["recommended_songs"] = []
        history.append(item)
    return history

