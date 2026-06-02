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
from core.zones import ZoneManager

# Единая папка для медиа тревог (скриншоты + видеоклипы).
# Раньше пути расходились (создавалась 'detections', а писалось в 'data/detections'),
# из-за чего скриншоты/клипы терялись. Теперь всё в одной папке 'detections'.
MEDIA_DIR = 'detections'

# ─── создаём нужные папки ───────────────────────────────────────────────────
for _d in ('templates', MEDIA_DIR, 'static'):
    os.makedirs(_d, exist_ok=True)

# ─── глобальные переменные ──────────────────────────────────────────────────
frame_buffer = None            # последний кадр для MJPEG-стрима
current_detections: list = []
current_class_counts: dict = {}
current_zone_stats: list = []  # живые метрики по зонам
detection_fps: int = 0
running: bool = True
video_clip_buffer: VideoClipBuffer | None = None

# track_id, по которым уже создавались ГЛОБАЛЬНЫЕ события (не дублируем)
_alerted_appear: set = set()
_alerted_long:   set = set()
# Время последней глобальной тревоги каждого типа — антидребезг
_last_global_event: dict = {}

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

# Менеджер зон срабатывания — основной источник «осмысленных» тревог.
zone_manager = ZoneManager(
    zones=SETTINGS.get('zones', []),
    cooldown_default=SETTINGS.get('event_cooldown_seconds', 15),
)

# ─── сохранение события-тревоги ─────────────────────────────────────────────

def save_event(event_type: str, obj: dict, frame, zone: str | None = None,
               save_media: bool = True):
    """
    Сохраняет тревогу: скриншот → БД → видеоклип → Telegram.
    save_media=False → пишем только строку в БД, без скриншота/клипа
    (используется, если у зоны отключено сохранение медиа).
    """
    class_name = obj.get('class')
    track_id   = obj.get('track_id')
    confidence = obj.get('confidence')

    # 1. Скриншот (только для «настоящих» тревог и если включено в настройках)
    screenshot_path = None
    if save_media and SETTINGS.get('save_screenshots', True):
        fname = f"shot_{time.strftime('%Y%m%d_%H%M%S')}_id{track_id}.jpg"
        path  = os.path.join(MEDIA_DIR, fname)
        try:
            cv2.imwrite(path, frame)
            screenshot_path = f'{MEDIA_DIR}/{fname}'
        except Exception as e:
            print(f'[Событие] Не удалось сохранить скриншот: {e}')

    # 2. Запись в SQLite (зона попадает в отдельную колонку для отчётности)
    event_id = database.add_event(
        event_type=event_type,
        class_name=class_name,
        track_id=track_id,
        confidence=confidence,
        zone=zone,
        screenshot=screenshot_path,
    )

    # 3. Видеоклип (асинхронно — внутри VideoClipBuffer)
    if save_media and SETTINGS.get('save_video_clips', True) and video_clip_buffer:
        def _on_clip_ready(path: str):
            database.update_event_clip(event_id, path.replace('\\', '/'))
        video_clip_buffer.start_clip(on_complete=_on_clip_ready)

    # 4. Telegram
    if SETTINGS.get('telegram_enabled', False):
        token   = SETTINGS.get('telegram_token', '')
        chat_id = SETTINGS.get('telegram_chat_id', '')
        zone_line = f'Зона: {zone}\n' if zone else ''
        caption = (f'🚨 {event_type}\n'
                   f'{zone_line}'
                   f'Объект: {class_name} (ID {track_id})\n'
                   f'Уверенность: {confidence}\n'
                   f'Время: {time.strftime("%H:%M:%S")}')
        if screenshot_path and os.path.exists(screenshot_path):
            telegram_notify.send_photo(token, chat_id, screenshot_path, caption)
        else:
            telegram_notify.send_message(token, chat_id, caption)

    print(f'[Событие] {event_type} | зона={zone} | {class_name} '
          f'ID:{track_id} conf:{confidence}')


def handle_events(frame, detections: list):
    """
    Создаёт тревоги. Главный механизм — ПРАВИЛА ЗОН: тревога возникает только в
    заданной области кадра, поэтому проезжающие мимо машины больше не засыпают
    папки скриншотами. Глобальные тревоги (на весь кадр) по умолчанию выключены
    и служат запасным вариантом, когда зоны ещё не настроены.
    """
    now = time.time()
    h, w = frame.shape[:2]

    # 1. Тревоги по зонам
    for ev in zone_manager.process(detections, w, h, now):
        save_event(ev['event_type'], ev['obj'], frame,
                   zone=ev['zone'], save_media=ev['save_media'])

    # 2. Глобальные тревоги — только если оператор явно их включил
    cooldown = SETTINGS.get('event_cooldown_seconds', 15)
    alert_conf = SETTINGS.get('alert_confidence', 0.7)
    appear_on  = SETTINGS.get('alert_on_appearance', False)
    long_on    = SETTINGS.get('global_long_stay_alerts', False)

    def _cooldown_ok(key: str) -> bool:
        if now - _last_global_event.get(key, 0) < cooldown:
            return False
        _last_global_event[key] = now
        return True

    if not (appear_on or long_on):
        return

    for obj in detections:
        tid = obj.get('track_id')
        if tid is None:
            continue
        if appear_on and tid not in _alerted_appear and \
                obj['confidence'] >= alert_conf and _cooldown_ok('appear'):
            _alerted_appear.add(tid)
            save_event('Появление объекта', obj, frame)
        if long_on and obj.get('long_stay') and tid not in _alerted_long and \
                _cooldown_ok('long'):
            _alerted_long.add(tid)
            save_event('Долгое нахождение', obj, frame)


# ─── поток захвата и обработки видео ────────────────────────────────────────

def capture_loop():
    global frame_buffer, current_detections, current_class_counts
    global current_zone_stats, detection_fps, running, video_clip_buffer

    source = _parse_video_source(SETTINGS.get('video_source', 'test_videos/test_video3.mp4'))
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
        out_dir=MEDIA_DIR,
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

        # Рисуем зоны поверх кадра — они видны в потоке и попадают в скриншоты.
        zone_manager.draw(processed)

        current_detections  = dets
        current_class_counts = counts

        if detector.enabled:
            handle_events(processed, dets)        # тревоги по зонам/глобальные
            video_clip_buffer.add_frame(processed)
            current_zone_stats = zone_manager.get_live()

        frame_buffer = processed

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
        'zones':             current_zone_stats,   # живые метрики по зонам
    }


@app.post('/api/detection/start')
async def api_start():
    _alerted_appear.clear()
    _alerted_long.clear()
    _last_global_event.clear()
    zone_manager.reset_runtime()
    detector.set_enabled(True)
    return {'status': 'started'}


@app.post('/api/detection/stop')
async def api_stop():
    global current_zone_stats
    detector.set_enabled(False)
    zone_manager.reset_runtime()           # обнуляем живые счётчики зон
    current_zone_stats = zone_manager.get_live()
    return {'status': 'stopped'}


# ─── API: события ────────────────────────────────────────────────────────────

@app.get('/api/events')
async def api_events(
    page: int = 1,
    per_page: int = 10,
    event_type: str = '',
    date_from: str = '',
    date_to: str = '',
    zone: str = '',
):
    et    = event_type or None
    df    = date_from  or None
    dt    = date_to    or None
    zn    = zone       or None
    total = database.count_events(et, df, dt, zn)
    items = database.get_events(
        limit=per_page,
        offset=(page - 1) * per_page,
        event_type=et, date_from=df, date_to=dt, zone=zn,
    )
    return {'events': items, 'total': total, 'page': page, 'per_page': per_page}


@app.get('/api/events/zone_report')
async def api_zone_report(date_from: str = '', date_to: str = ''):
    """Отдельная отчётность по зонам (сводка тревог по каждой зоне)."""
    return {'report': database.zone_report(date_from or None, date_to or None)}


@app.post('/api/events/clear')
async def api_events_clear():
    database.clear_events()
    return {'status': 'cleared'}


# ─── API: зоны срабатывания ──────────────────────────────────────────────────

@app.get('/api/zones')
async def api_get_zones():
    """Текущие зоны (для рисования на плеере и редактирования правил)."""
    return {'zones': SETTINGS.get('zones', [])}


@app.post('/api/zones')
async def api_save_zones(request: Request):
    """Сохраняет список зон и сразу применяет его к менеджеру зон (без рестарта)."""
    global SETTINGS
    data = await request.json()
    zones = data.get('zones', [])
    SETTINGS = config.save_settings({'zones': zones})
    zone_manager.update_zones(zones)
    return {'status': 'saved', 'zones': zones}


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
    zone_manager.update_zones(SETTINGS.get('zones', []))
    zone_manager.cooldown_default = SETTINGS.get('event_cooldown_seconds', 15)
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
