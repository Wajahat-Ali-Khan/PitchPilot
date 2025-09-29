import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict

class Database:
    def __init__(self, db_path: str = "pitchpilot.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pitch_decks (
                id TEXT PRIMARY KEY,
                idea_text TEXT,
                slides_json TEXT,
                pdf_path TEXT,
                created_at TEXT
            )
            """
        )
        self.conn.commit()

    def save_deck(self, deck_id: str, idea_text: str, slides_json: str, pdf_path: str, created_at: datetime):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO pitch_decks (id, idea_text, slides_json, pdf_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (deck_id, idea_text, slides_json, pdf_path, created_at.isoformat()),
        )
        self.conn.commit()

    def list_recent(self, limit: int = 10) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, idea_text, pdf_path, created_at FROM pitch_decks ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_deck(self, deck_id: str) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM pitch_decks WHERE id = ?", (deck_id,))
        row = cur.fetchone()
        return dict(row) if row else None
