"""SQLite-backed recommendation memory."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

from app.config.settings import DATABASE_PATH
from app.database.db import get_connection


def create_personalization_tables(db_path: str | Path = DATABASE_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                artist TEXT,
                mood TEXT,
                genre TEXT,
                language TEXT,
                action TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def remember_feedback(
    title: str,
    artist: str = "",
    mood: str = "",
    genre: str = "",
    language: str = "",
    action: str = "liked",
    db_path: str | Path = DATABASE_PATH,
) -> None:
    create_personalization_tables(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO recommendation_feedback (title, artist, mood, genre, language, action)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (title, artist, mood, genre, language, action),
        )


def fetch_feedback(limit: int = 100, db_path: str | Path = DATABASE_PATH) -> List[Dict[str, object]]:
    create_personalization_tables(db_path)
    with get_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM recommendation_feedback ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]

