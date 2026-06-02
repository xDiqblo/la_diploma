"""
main_web.py — локальный веб-сервер (FastAPI) для системы видеонаблюдения АЗС.
Запуск: python main_web.py
Адрес: http://localhost:8000
"""
import cv2
import time
import threading
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from core import telegram_notify, database
from core.detector import Detector
from core.video_buffer import VideoClipBuffer
from core.cloud_sync import CloudSync

# ─── создаём нужные папки ───────────────────────────────────────────────────
for _d in ('templates', 'detections', 'static'):
    os.makedirs(_d, exist_ok=True)

# ─── глобальные переменные ──────────────────────────────────────────────────
frame_buffer = None            # последний кадр для MJPEG-стрима
current_detections: list = []
current_class_counts: dict = {}
detection_fps: int = 0
running: bool = True
video_clip_buffer: VideoClipBuffer | None = None

# track_id, по которым уже создавались события (не дублируем)
_alerted_appear: set = set()
_alerted_long:   set = set()

# ─── загрузка настроек и инициализация зависимостей ─────────────────────────
SETTINGS = config.load_settings()


def _parse_video_source(src: str):
    """Строка '0' → int(0) (USB-камера), иначе — RTSP/путь к файлу."""
    try:
        return int(src)
    except ValueError:
        return src


detector = Detector(
    model_path='model/yolov8n.pt',
    confidence_threshold=SETTINGS['confidence_threshold'],
    frame_skip=SETTINGS['frame_skip'],
    active_classes=SETTINGS['active_classes'],
    long_stay_seconds=SETTINGS['long_stay_seconds'],
)

cloud_sync = CloudSync(
    cloud_url=SETTINGS.get('cloud_url', 'http://localhost:8001'),
    api_key=SETTINGS.get('cloud_api_key', ''),
    enabled=SETTINGS.get('cloud_sync_enabled', False),
)

# ─── сохранение события-тревоги ─────────────────────────────────────────────

def save_event(event_type: str, obj: dict, frame):
    """Скриншот → БД → видеоклип → Telegram. Всё обёрнуто в try/except."""
    class_name = obj['class']
    track_id   = obj['track_id']
    confidence = obj['confidence']

    # 1. Скриншот
    screenshot_path = None
    if SETTINGS.get('save_screenshots', True):
        fname = f"shot_{time.strftime('%Y%m%d_%H%M%S')}_id{track_id}.jpg"
        path  = os.path.join('data/detections', fname)
        try:
            cv2.imwrite(path, frame)
            screenshot_path = f'data/detections/{fname}'
        except Exception as e:
            print(f'[Событие] Не удалось сохранить скриншот: {e}')

    # 2. Запись в SQLite
    event_id = database.add_event(
        event_type=event_type,
        class_name=class_name,
        track_id=track_id,
        confidence=confidence,
        screenshot=screenshot_path,
    )

    # 3. Видеоклип (асинхронно — внутри VideoClipBuffer)
    if SETTINGS.get('save_video_clips', True) and video_clip_buffer:
        def _on_clip_ready(path: str):
            database.update_event_clip(event_id, path.replace('\\', '/'))
        video_clip_buffer.start_clip(on_complete=_on_clip_ready)

    # 4. Telegram
    if SETTINGS.get('telegram_enabled', False):
        token   = SETTINGS.get('telegram_token', '')
        chat_id = SETTINGS.get('telegram_chat_id', '')
        caption = (f'🚨 {event_type}\n'
                   f'Объект: {class_name} (ID {track_id})\n'
                   f'Уверенность: {confidence}\n'
                   f'Время: {time.strftime("%H:%M:%S")}')
        if screenshot_path and os.path.exists(screenshot_path):
            telegram_notify.send_photo(token, chat_id, screenshot_path, caption)
        else:
            telegram_notify.send_message(token, chat_id, caption)

    print(f'[Событие] {event_type} | {class_name} ID:{track_id} conf:{confidence}')


def handle_events(frame, detections: list):
    """Анализирует детекции и при необходимости создаёт тревоги."""
    alert_conf = SETTINGS.get('alert_confidence', 0.7)
    for obj in detections:
        tid = obj.get('track_id')
        if tid is None:
            continue
        # Тревога 1: уверенное появление нового объекта
        if tid not in _alerted_appear and obj['confidence'] >= alert_conf:
            _alerted_appear.add(tid)
            save_event('Появление объекта', obj, frame)
        # Тревога 2: объект слишком долго в кадре
        if obj.get('long_stay') and tid not in _alerted_long:
            _alerted_long.add(tid)
            save_event('Долгое нахождение', obj, frame)


# ─── поток захвата и обработки видео ────────────────────────────────────────

def capture_loop():
    global frame_buffer, current_detections, current_class_counts
    global detection_fps, running, video_clip_buffer

    source = _parse_video_source(SETTINGS.get('video_source', '0'))
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f'[ОШИБКА] Не удалось открыть источник видео: {source}')
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay = 1.0 / video_fps
    print(f'[Камера] Источник: {source}  FPS: {video_fps:.1f}')

    # Кольцевой буфер видео
    video_clip_buffer = VideoClipBuffer(
        fps=int(video_fps),
        pre_seconds=SETTINGS.get('clip_pre_seconds', 10),
        post_seconds=SETTINGS.get('clip_post_seconds', 10),
        out_dir='data/detections',
    )

    frame_count   = 0
    fps_timer     = time.time()
    last_frame_t  = time.time()

    while running:
        # Контроль скорости (важно при воспроизведении видеофайла, не RTSP)
        elapsed = time.time() - last_frame_t
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # зацикливаем видеофайл
            continue
        last_frame_t = time.time()

        processed, dets, counts = detector.process_frame(frame)
        frame_buffer        = processed
        current_detections  = dets
        current_class_counts = counts

        if detector.enabled:
            video_clip_buffer.add_frame(processed)
            handle_events(processed, dets)

        # Подсчёт FPS
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            detection_fps = frame_count
            frame_count   = 0
            fps_timer     = time.time()

    cap.release()


# ─── lifespan FastAPI ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global running
    database.init_db()
    running = True
    t = threading.Thread(target=capture_loop, daemon=True, name='capture')
    t.start()
    cloud_sync.start()
    print('=' * 55)
    print('  Веб-приложение камеры АЗС (локальный хост)')
    print('  http://localhost:8000')
    print('=' * 55)
    yield
    running = False
    cloud_sync.stop()
    t.join(timeout=3)
    print('[Сервер] Остановлен')


# ─── приложение ─────────────────────────────────────────────────────────────

app = FastAPI(title='Камера АЗС', lifespan=lifespan)
app.mount('/detections', StaticFiles(directory='detections'), name='detections')
app.mount('/static',     StaticFiles(directory='static'),     name='static')


# ─── MJPEG стрим ─────────────────────────────────────────────────────────────

def _mjpeg_gen():
    while running:
        if frame_buffer is not None:
            ok, jpeg = cv2.imencode('.jpg', frame_buffer,
                                    [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + jpeg.tobytes() + b'\r\n')
        else:
            time.sleep(0.02)


@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(
        _mjpeg_gen(),
        media_type='multipart/x-mixed-replace; boundary=frame',
    )


# ─── HTML страницы ────────────────────────────────────────────────────────────

def _read_html(name: str) -> str:
    path = os.path.join('templates', name)
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f'<h1>Файл templates/{name} не найден</h1>'


@app.get('/',         response_class=HTMLResponse)
async def page_index():    return HTMLResponse(_read_html('index.html'))

@app.get('/events',   response_class=HTMLResponse)
async def page_events():   return HTMLResponse(_read_html('events.html'))

@app.get('/settings', response_class=HTMLResponse)
async def page_settings(): return HTMLResponse(_read_html('settings.html'))


# ─── API: детекция ───────────────────────────────────────────────────────────

@app.get('/api/detections')
async def api_detections():
    return {
        'detection_enabled': detector.enabled,
        'fps':               detection_fps,
        'objects':           current_detections,
        'class_counts':      current_class_counts,
    }


@app.post('/api/detection/start')
async def api_start():
    _alerted_appear.clear()
    _alerted_long.clear()
    detector.set_enabled(True)
    return {'status': 'started'}


@app.post('/api/detection/stop')
async def api_stop():
    detector.set_enabled(False)
    return {'status': 'stopped'}


# ─── API: события ────────────────────────────────────────────────────────────

@app.get('/api/events')
async def api_events(
    page: int = 1,
    per_page: int = 10,
    event_type: str = '',
    date_from: str = '',
    date_to: str = '',
):
    et    = event_type or None
    df    = date_from  or None
    dt    = date_to    or None
    total = database.count_events(et, df, dt)
    items = database.get_events(
        limit=per_page,
        offset=(page - 1) * per_page,
        event_type=et, date_from=df, date_to=dt,
    )
    return {'events': items, 'total': total, 'page': page, 'per_page': per_page}


@app.post('/api/events/clear')
async def api_events_clear():
    database.clear_events()
    return {'status': 'cleared'}


# ─── API: настройки ──────────────────────────────────────────────────────────

@app.get('/api/settings')
async def api_get_settings():
    return SETTINGS


@app.post('/api/settings')
async def api_save_settings(request: Request):
    global SETTINGS
    data = await request.json()
    SETTINGS = config.save_settings(data)
    # Применяем к детектору без перезапуска
    detector.update_settings(
        confidence_threshold=SETTINGS['confidence_threshold'],
        frame_skip=SETTINGS['frame_skip'],
        active_classes=SETTINGS['active_classes'],
        long_stay_seconds=SETTINGS['long_stay_seconds'],
    )
    cloud_sync.update(
        cloud_url=SETTINGS.get('cloud_url', 'http://localhost:8001'),
        api_key=SETTINGS.get('cloud_api_key', ''),
        enabled=SETTINGS.get('cloud_sync_enabled', False),
    )
    return {'status': 'saved', 'settings': SETTINGS}


# ─── API: Telegram тест ──────────────────────────────────────────────────────

@app.post('/api/telegram/test')
async def api_telegram_test():
    ok, msg = telegram_notify.test_connection(
        SETTINGS.get('telegram_token', ''),
        SETTINGS.get('telegram_chat_id', ''),
    )
    return JSONResponse({'ok': ok, 'message': msg})


# ─── точка входа ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
