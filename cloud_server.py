"""
cloud_server.py — облачный сервер (эмуляция на localhost:8001).
Запуск: python cloud_server.py
API:           http://localhost:8001/api/...
Тонкий клиент: http://localhost:8001/cloud
"""
import os
import sqlite3
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, Header, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

CLOUD_DB   = 'cloud_events.db'
CLOUD_STOR = 'cloud_storage'
API_KEY    = 'secret-key-123'   # должен совпадать с cloud_api_key в settings.json

os.makedirs(CLOUD_STOR, exist_ok=True)
os.makedirs('templates',  exist_ok=True)

_lock = threading.Lock()


# ─── БД ───────────────────────────────────────────────────────────────────────

def _conn():
    c = sqlite3.connect(CLOUD_DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def cloud_init_db():
    with _lock:
        c = _conn()
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY,
                received_at TEXT NOT NULL,
                timestamp   TEXT,
                event_type  TEXT,
                class_name  TEXT,
                track_id    INTEGER,
                confidence  REAL,
                screenshot  TEXT
            )
        """)
        c.commit()
        c.close()


def cloud_add(ev: dict) -> bool:
    """Идемпотентная вставка (повторные попытки не создают дубли)."""
    with _lock:
        c = _conn()
        c.execute("""
            INSERT OR IGNORE INTO events
            (id, received_at, timestamp, event_type, class_name, track_id, confidence)
            VALUES (?,?,?,?,?,?,?)""",
            (ev.get('id'),
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             ev.get('timestamp'), ev.get('event_type'),
             ev.get('class_name'), ev.get('track_id'), ev.get('confidence')),
        )
        c.commit()
        changed = c.execute('SELECT changes()').fetchone()[0]
        c.close()
        return changed > 0


def cloud_set_screenshot(event_id: int, path: str):
    with _lock:
        c = _conn()
        c.execute("UPDATE events SET screenshot=? WHERE id=?", (path, event_id))
        c.commit()
        c.close()


def cloud_get_events(limit=20, offset=0, event_type=None,
                     date_from=None, date_to=None):
    with _lock:
        c = _conn()
        cl, pr = [], []
        if event_type: cl.append("event_type=?"); pr.append(event_type)
        if date_from:  cl.append("timestamp>=?"); pr.append(date_from)
        if date_to:    cl.append("timestamp<=?"); pr.append(date_to + ' 23:59:59')
        w = ("WHERE " + " AND ".join(cl)) if cl else ""
        rows  = c.execute(f"SELECT * FROM events {w} ORDER BY id DESC LIMIT ? OFFSET ?",
                          (*pr, limit, offset)).fetchall()
        total = c.execute(f"SELECT COUNT(*) FROM events {w}", pr).fetchone()[0]
        c.close()
        return [dict(r) for r in rows], total


def cloud_stats():
    """Число событий по дням (для графика)."""
    with _lock:
        c = _conn()
        rows = c.execute("""
            SELECT substr(timestamp,1,10) AS day, COUNT(*) AS cnt
            FROM events
            GROUP BY day ORDER BY day DESC LIMIT 30
        """).fetchall()
        c.close()
        return [dict(r) for r in rows]


# ─── авторизация ─────────────────────────────────────────────────────────────

def _check_key(key: str | None):
    if key != API_KEY:
        raise HTTPException(403, 'Неверный API-ключ')


# ─── lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cloud_init_db()
    print('=' * 55)
    print('  Облачный сервер АЗС')
    print('  API:           http://localhost:8001')
    print('  Тонкий клиент: http://localhost:8001/cloud')
    print('=' * 55)
    yield


app = FastAPI(title='АЗС Облако', lifespan=lifespan)
app.mount('/cloud_storage', StaticFiles(directory=CLOUD_STOR), name='cloud_storage')


# ─── вспомогательное чтение HTML ─────────────────────────────────────────────

def _html(name: str) -> str:
    path = os.path.join('templates', name)
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f'<h1>Файл templates/{name} не найден</h1>'


# ─── тонкий клиент ───────────────────────────────────────────────────────────

@app.get('/cloud', response_class=HTMLResponse)
async def cloud_ui():
    return HTMLResponse(_html('cloud_index.html'))


# ─── API для локального хоста (приём событий) ────────────────────────────────

@app.post('/api/events', status_code=201)
async def recv_event(request: Request,
                     x_api_key: str = Header(default=None)):
    _check_key(x_api_key)
    cloud_add(await request.json())
    return {'status': 'ok'}


@app.post('/api/events/{event_id}/screenshot')
async def recv_screenshot(event_id: int,
                          file: UploadFile = File(...),
                          x_api_key: str = Header(default=None)):
    _check_key(x_api_key)
    dest = os.path.join(CLOUD_STOR, f'shot_{event_id}.jpg')
    with open(dest, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    cloud_set_screenshot(event_id, f'cloud_storage/shot_{event_id}.jpg')
    return {'status': 'ok'}


# ─── API для тонкого клиента ─────────────────────────────────────────────────

@app.get('/api/cloud/events')
async def api_cloud_events(
    page: int = 1, per_page: int = 10,
    event_type: str = '', date_from: str = '', date_to: str = '',
):
    items, total = cloud_get_events(
        limit=per_page, offset=(page - 1) * per_page,
        event_type=event_type or None,
        date_from=date_from or None,
        date_to=date_to or None,
    )
    return {'events': items, 'total': total, 'page': page, 'per_page': per_page}


@app.get('/api/cloud/stats')
async def api_cloud_stats():
    return {'stats': cloud_stats()}


# ─── заглушка авторизации ────────────────────────────────────────────────────

@app.post('/api/cloud/login')
async def api_login(request: Request):
    data = await request.json()
    if data.get('username') == 'admin' and data.get('password') == 'admin':
        return {'ok': True, 'token': 'demo-token'}
    return JSONResponse({'ok': False, 'message': 'Неверный логин или пароль'}, 401)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)
