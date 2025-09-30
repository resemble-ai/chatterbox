"""Simple SQLite storage for chat history and manual mode mappings."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional, Tuple


ChatRecord = Tuple[str, str]


class ChatDatabase:
    """Tiny helper around SQLite for persisting chats."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._setup()

    def close(self) -> None:
        self._conn.close()

    def _setup(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                chat_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                manual_mode INTEGER DEFAULT 0,
                voice_enabled INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS manual_queue (
                admin_message_id INTEGER PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                user_message_id INTEGER NOT NULL,
                info_message_id INTEGER,
                FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute("PRAGMA table_info(manual_queue)")
        columns = {row[1] for row in cur.fetchall()}
        if "info_message_id" not in columns:
            cur.execute("ALTER TABLE manual_queue ADD COLUMN info_message_id INTEGER")
        cur.close()
        self._conn.commit()

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        cur = self._conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def ensure_chat(self, chat_id: int, user_id: Optional[int] = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO chats(chat_id, user_id) VALUES (?, ?)",
                (chat_id, user_id),
            )
            if user_id is not None:
                cur.execute("UPDATE chats SET user_id = ? WHERE chat_id = ?", (user_id, chat_id))
        self._conn.commit()

    def set_manual_mode(self, chat_id: int, enabled: bool) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE chats SET manual_mode = ? WHERE chat_id = ?", (int(enabled), chat_id))
            cur.execute("DELETE FROM manual_queue WHERE chat_id = ?", (chat_id,))
        self._conn.commit()

    def is_manual_mode(self, chat_id: int) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT manual_mode FROM chats WHERE chat_id = ?", (chat_id,))
            row = cur.fetchone()
        return bool(row[0]) if row else False

    def set_voice_enabled(self, chat_id: int, enabled: bool) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE chats SET voice_enabled = ? WHERE chat_id = ?", (int(enabled), chat_id))
        self._conn.commit()

    def is_voice_enabled(self, chat_id: int) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT voice_enabled FROM chats WHERE chat_id = ?", (chat_id,))
            row = cur.fetchone()
        return bool(row[0]) if row else False

    def add_message(self, chat_id: int, role: str, content: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO messages(chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, role, content),
            )
        self._conn.commit()

    def get_history(self, chat_id: int, limit: int = 20) -> List[ChatRecord]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT role, content FROM messages
                WHERE chat_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (chat_id, limit),
            )
            rows = cur.fetchall()
        return [(role, content) for role, content in rows]

    def clear_chat(self, chat_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            cur.execute("DELETE FROM manual_queue WHERE chat_id = ?", (chat_id,))
        self._conn.commit()

    def register_manual_forward(
        self,
        *,
        admin_message_id: int,
        chat_id: int,
        user_id: int,
        user_message_id: int,
        info_message_id: Optional[int] = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO manual_queue(admin_message_id, chat_id, user_id, user_message_id, info_message_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (admin_message_id, chat_id, user_id, user_message_id, info_message_id),
            )
        self._conn.commit()


    def resolve_manual_reply(self, admin_message_id: int) -> Optional[Tuple[int, int, int, Optional[int]]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT chat_id, user_id, user_message_id, info_message_id FROM manual_queue WHERE admin_message_id = ?",
                (admin_message_id,),
            )
            row = cur.fetchone()
        return tuple(row) if row else None

