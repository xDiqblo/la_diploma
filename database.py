"""
database.py — хранение событий (тревог) в базе данных SQLite.
Каждое событие: время, тип аномалии, класс объекта, уверенность,
путь к скриншоту и к видеофрагменту.
"""
import sqlite3
import threading
from datetime import datetime

DB_FILE = 'events.db'

# SQLite из разных потоков: используем check_same_thread=False + блокировку
_lock = threading.Lock()


def _connect():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # доступ к колонкам по имени
    return conn


def init_db():
    """Создаёт таблицу событий, если её ещё нет. Вызывается при старте сервера."""
    with _lock:
        conn = _connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                class_name  TEXT,
                track_id    INTEGER,
                confidence  REAL,
                screenshot  TEXT,
                video_clip  TEXT
            )
        """)
        conn.commit()
        conn.close()


def add_event(event_type, class_name=None, track_id=None, confidence=None,
              screenshot=None, video_clip=None):
    """Добавляет событие в БД и возвращает его id."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with _lock:
        conn = _connect()
        cur = conn.execute(
            """INSERT INTO events
               (timestamp, event_type, class_name, track_id, confidence, screenshot, video_clip)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, event_type, class_name, track_id, confidence, screenshot, video_clip)
        )
        conn.commit()
        event_id = cur.lastrowid
        conn.close()
        return event_id


def update_event_clip(event_id, video_clip):
    """Дописывает путь к видеофрагменту (клип готовится асинхронно после события)."""
    with _lock:
        conn = _connect()
        conn.execute("UPDATE events SET video_clip = ? WHERE id = ?",
                     (video_clip, event_id))
        conn.commit()
        conn.close()


def get_events(limit=100):
    """Возвращает список последних событий (новые сверху) как список словарей."""
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]


def clear_events():
    """Удаляет все события (кнопка «Очистить» на странице событий)."""
    with _lock:
        conn = _connect()
        conn.execute("DELETE FROM events")
        conn.commit()
        conn.close()
