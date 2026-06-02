"""
database.py — локальная SQLite-база событий (тревог).

Каждая запись: время, тип аномалии, класс объекта, уверенность,
пути к скриншоту и видеофрагменту, флаг синхронизации с облаком.
"""
import sqlite3
import threading
from datetime import datetime

DB_FILE = '../data/events.db'
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_FILE, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Создаёт таблицу событий при старте. Безопасно вызывать повторно."""
    with _lock:
        c = _conn()
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,
                class_name  TEXT,
                track_id    INTEGER,
                confidence  REAL,
                screenshot  TEXT,
                video_clip  TEXT,
                synced      INTEGER DEFAULT 0  -- 0 = ещё не отправлено в облако
            )
        """)
        c.commit()
        c.close()


def add_event(event_type, class_name=None, track_id=None,
              confidence=None, screenshot=None, video_clip=None) -> int:
    """Добавляет событие, возвращает его id."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with _lock:
        c = _conn()
        cur = c.execute(
            """INSERT INTO events
               (timestamp, event_type, class_name, track_id, confidence, screenshot, video_clip)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ts, event_type, class_name, track_id, confidence, screenshot, video_clip),
        )
        c.commit()
        eid = cur.lastrowid
        c.close()
        return eid


def update_event_clip(event_id: int, video_clip: str):
    """Дописывает путь к видеоклипу (клип готовится асинхронно)."""
    with _lock:
        c = _conn()
        c.execute("UPDATE events SET video_clip=? WHERE id=?", (video_clip, event_id))
        c.commit()
        c.close()


def mark_synced(event_id: int):
    """Помечает событие как отправленное в облако."""
    with _lock:
        c = _conn()
        c.execute("UPDATE events SET synced=1 WHERE id=?", (event_id,))
        c.commit()
        c.close()


def get_events(limit=200, offset=0, event_type=None,
               date_from=None, date_to=None) -> list[dict]:
    """Список событий с фильтрацией и пагинацией (новые сверху)."""
    with _lock:
        c = _conn()
        clauses, params = [], []
        if event_type:
            clauses.append("event_type=?"); params.append(event_type)
        if date_from:
            clauses.append("timestamp>=?"); params.append(date_from)
        if date_to:
            clauses.append("timestamp<=?"); params.append(date_to + ' 23:59:59')
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = c.execute(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        c.close()
        return [dict(r) for r in rows]


def count_events(event_type=None, date_from=None, date_to=None) -> int:
    """Число событий (для пагинации)."""
    with _lock:
        c = _conn()
        clauses, params = [], []
        if event_type:
            clauses.append("event_type=?"); params.append(event_type)
        if date_from:
            clauses.append("timestamp>=?"); params.append(date_from)
        if date_to:
            clauses.append("timestamp<=?"); params.append(date_to + ' 23:59:59')
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        row = c.execute(f"SELECT COUNT(*) FROM events {where}", params).fetchone()
        c.close()
        return row[0]


def get_unsynced(limit=50) -> list[dict]:
    """События, ещё не отправленные в облако."""
    with _lock:
        c = _conn()
        rows = c.execute(
            "SELECT * FROM events WHERE synced=0 ORDER BY id LIMIT ?", (limit,)
        ).fetchall()
        c.close()
        return [dict(r) for r in rows]


def clear_events():
    """Удаляет все события."""
    with _lock:
        c = _conn()
        c.execute("DELETE FROM events")
        c.commit()
        c.close()
