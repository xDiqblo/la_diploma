"""
database.py — локальная SQLite-база событий (тревог).

Каждая запись: время, тип аномалии, класс объекта, уверенность,
пути к скриншоту и видеофрагменту, флаг синхронизации с облаком.
"""
import sqlite3
import threading
from datetime import datetime

# Путь к БД относительно корня проекта (откуда запускается main_web.py).
# Раньше здесь было '../data/events.db' — путь уводил на уровень выше проекта,
# из-за чего события писались «мимо» и терялись. Исправлено на корректный путь.
DB_FILE = 'data/events.db'
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
                zone        TEXT,              -- имя зоны срабатывания (NULL = весь кадр)
                screenshot  TEXT,
                video_clip  TEXT,
                synced      INTEGER DEFAULT 0  -- 0 = ещё не отправлено в облако
            )
        """)
        # Миграция: если БД создана старой версией без колонки zone — добавляем.
        cols = {r[1] for r in c.execute("PRAGMA table_info(events)").fetchall()}
        if 'zone' not in cols:
            c.execute("ALTER TABLE events ADD COLUMN zone TEXT")
        c.commit()
        c.close()


def add_event(event_type, class_name=None, track_id=None,
              confidence=None, zone=None, screenshot=None, video_clip=None) -> int:
    """Добавляет событие, возвращает его id."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with _lock:
        c = _conn()
        cur = c.execute(
            """INSERT INTO events
               (timestamp, event_type, class_name, track_id, confidence, zone, screenshot, video_clip)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, event_type, class_name, track_id, confidence, zone, screenshot, video_clip),
        )
        c.commit()
        eid = cur.lastrowid
        c.close()
        return eid


def mark_synced(event_id: int):
    """Помечает событие как отправленное в облако."""
    with _lock:
        c = _conn()
        c.execute("UPDATE events SET synced=1 WHERE id=?", (event_id,))
        c.commit()
        c.close()


def _build_where(event_type, date_from, date_to, zone):
    """Собирает WHERE-условие и параметры для фильтров (общая часть запросов)."""
    clauses, params = [], []
    if event_type:
        clauses.append("event_type=?"); params.append(event_type)
    if date_from:
        clauses.append("timestamp>=?"); params.append(date_from)
    if date_to:
        clauses.append("timestamp<=?"); params.append(date_to + ' 23:59:59')
    if zone:
        clauses.append("zone=?"); params.append(zone)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def get_events(limit=200, offset=0, event_type=None,
               date_from=None, date_to=None, zone=None) -> list[dict]:
    """Список событий с фильтрацией и пагинацией (новые сверху)."""
    with _lock:
        c = _conn()
        where, params = _build_where(event_type, date_from, date_to, zone)
        rows = c.execute(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        c.close()
        return [dict(r) for r in rows]


def count_events(event_type=None, date_from=None, date_to=None, zone=None) -> int:
    """Число событий (для пагинации)."""
    with _lock:
        c = _conn()
        where, params = _build_where(event_type, date_from, date_to, zone)
        row = c.execute(f"SELECT COUNT(*) FROM events {where}", params).fetchone()
        c.close()
        return row[0]


def zone_report(date_from=None, date_to=None) -> list[dict]:
    """
    Отчёт по зонам: сколько тревог дала каждая зона (и какого типа).
    Используется на странице «События» для отдельной отчётности по зонам.
    """
    with _lock:
        c = _conn()
        where, params = _build_where(None, date_from, date_to, None)
        rows = c.execute(f"""
            SELECT COALESCE(zone, 'Без зоны (весь кадр)') AS zone,
                   COUNT(*)                               AS total,
                   SUM(event_type LIKE 'Зона: вход%')     AS presence,
                   SUM(event_type LIKE 'Зона: задержка%') AS long_stay,
                   SUM(event_type LIKE 'Зона: скопление%')AS crowd
            FROM events {where}
            GROUP BY zone
            ORDER BY total DESC
        """, params).fetchall()
        c.close()
        return [dict(r) for r in rows]


def get_unsynced(limit=50, min_age_seconds=0) -> list[dict]:
    """События, ещё не отправленные в облако (новые сначала по возрастанию id)."""
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
