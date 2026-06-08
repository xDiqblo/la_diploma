"""
main_web.py — локальный веб-сервер (FastAPI) для системы видеонаблюдения АЗС.
Запуск: python main_web.py
Адрес: http://localhost:8000
"""

import cv2
import logging
import logging.handlers
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import config
from core import database
from core import classes as class_registry
from core import rules as rule_engine
from core import email_notify
from core import zone_store
from core.detector import Detector
from core.cloud_sync import CloudSync
from core.zones import ZoneManager
from core.rtsp_manager import RTSPManager, probe_source
from core.anomaly_detector import AnomalyDetector
from core.incremental_trainer import (
    HardExampleCollector,
    PseudoLabeler,
    IncrementalTrainer,
    DIR_PENDING,
    DIR_HARD,
    DIR_CONFIRMED,
)

# ---------------------------------------------------------------------------
# Логирование с ротацией
# ---------------------------------------------------------------------------

os.makedirs('logs', exist_ok=True)
_log_handler = logging.handlers.RotatingFileHandler(
    'logs/app.log', maxBytes=10 * 1024 * 1024, backupCount=3, encoding='utf-8'
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[_log_handler, logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Папки
# ---------------------------------------------------------------------------

MEDIA_DIR   = 'detections'
UPLOADS_DIR = 'data/uploads'   # временное хранилище загруженных видеофайлов

for _d in (
    'templates', MEDIA_DIR, 'static', UPLOADS_DIR,
    'data/hard_examples', 'data/pending_labels',
    'data/confirmed_labels', 'model/versions',
):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Глобальное состояние
# ---------------------------------------------------------------------------

frame_buffer: any          = None
current_detections: list   = []
current_class_counts: dict = {}
current_zone_stats: list   = []
detection_fps: int         = 0
running: bool              = True

_alerted_appear: set  = set()
_alerted_long:   set  = set()
_last_global_event: dict = {}

# ---------------------------------------------------------------------------
# Инициализация компонентов
# ---------------------------------------------------------------------------

SETTINGS = config.load_settings()


def _best_model_path() -> str:
    """Возвращает путь к лучшей доступной модели."""
    for p in ('model/best.pt', 'model/yolov8m.pt'):
        if os.path.exists(p):
            return p
    return 'model/yolov8m.pt'  # ultralytics скачает сама


detector = Detector(
    model_path=_best_model_path(),
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

# Зоны хранятся в отдельных файлах zones/zoneX.json (core/zone_store.py).
# Миграция: если файлов зон ещё нет, но в старом settings.json остались зоны —
# переносим их в новое хранилище один раз.
_stored_zones = zone_store.load_all()
if not _stored_zones and SETTINGS.get('zones'):
    zone_store.save_all(SETTINGS['zones'])
# При старте все зоны выключены (требование ТЗ) — оператор включает вручную.
_startup_zones = zone_store.load_all(reset_enabled=True)

zone_manager = ZoneManager(
    zones=_startup_zones,
    cooldown_default=SETTINGS.get('event_cooldown_seconds', 15),
)

anomaly_detector = AnomalyDetector(
    person_alone_timeout=60.0,
    person_alone_cooldown=60.0,
    person_long_stay_timeout=600.0,
    person_long_cooldown=300.0,
    car_alone_timeout=300.0,
    car_alone_cooldown=120.0,
)

rtsp_manager = RTSPManager(
    source=SETTINGS.get('video_source', '0'),
)

hard_collector = HardExampleCollector()
pseudo_labeler = PseudoLabeler()
trainer        = IncrementalTrainer(model_dir='model')

# ---------------------------------------------------------------------------
# Сохранение события
# ---------------------------------------------------------------------------

def save_event(event_type: str, obj: dict, frame,
               zone: str | None = None, save_media: bool = True):
    class_name = obj.get('class')
    track_id   = obj.get('track_id')
    confidence = obj.get('confidence')

    screenshot_path = None
    if save_media and SETTINGS.get('save_screenshots', True) and frame is not None:
        fname = f"shot_{time.strftime('%Y%m%d_%H%M%S')}_id{track_id}.jpg"
        path  = os.path.join(MEDIA_DIR, fname)
        try:
            cv2.imwrite(path, frame)
            screenshot_path = f'{MEDIA_DIR}/{fname}'
        except Exception as exc:
            logger.error('Не удалось сохранить скриншот: %s', exc)

    database.add_event(
        event_type=event_type,
        class_name=class_name,
        track_id=track_id,
        confidence=confidence,
        zone=zone,
        screenshot=screenshot_path,
    )

    # Уведомление по email на любую тревогу (зоны, правила, аномалии, глобальные).
    _notify_email(event_type, class_name, track_id, confidence, zone, screenshot_path)

    logger.info('Событие: %s | зона=%s | %s ID:%s conf:%s',
                event_type, zone, class_name, track_id, confidence)


# Состояние диспетчера тревог (антидребезг писем по типу+зоне)
_rule_fire_counts: dict = {}   # совместимость со сбросом в api_start/api_stop
_email_last_sent:  dict = {}   # (event_type, zone) -> время последнего письма


def _send_email_async(subject: str, body: str, screenshot: str | None):
    """Отправляет письмо в отдельном потоке, чтобы не блокировать захват кадров."""
    cfg = {k: SETTINGS.get(k) for k in (
        'email_enabled', 'email_smtp_host', 'email_smtp_port', 'email_use_tls',
        'email_login', 'email_password', 'email_from', 'email_to')}
    attach = screenshot if SETTINGS.get('email_attach_screenshot', True) else None

    def _worker():
        ok, msg = email_notify.send_alert(cfg, subject, body, attach)
        if ok:
            logger.info('Тревога отправлена на email')
        else:
            logger.warning('Тревога по email не отправлена: %s', msg)

    threading.Thread(target=_worker, daemon=True).start()


def _notify_email(event_type, class_name, track_id, confidence, zone, screenshot):
    """
    Отправляет письмо-тревогу (если включено), с антидребезгом по типу+зоне,
    чтобы один и тот же тип события не слал письма каждый кадр.
    """
    if not SETTINGS.get('email_enabled'):
        return

    now = time.time()
    cooldown = max(10, int(SETTINGS.get('event_cooldown_seconds', 15)))
    key = (event_type, zone or '')
    if now - _email_last_sent.get(key, 0) < cooldown:
        return
    _email_last_sent[key] = now

    zone_line = f'Зона: {zone}\n' if zone else 'Зона: весь кадр\n'
    conf_line = f'Уверенность: {confidence}\n' if confidence is not None else ''
    subject = f'Тревога АЗС: {event_type}'
    body = (
        f'Сработала тревога системы видеонаблюдения АЗС.\n\n'
        f'Тип: {event_type}\n'
        f'{zone_line}'
        f'Объект: {class_name or "—"}'
        f'{f" (ID {track_id})" if track_id is not None else ""}\n'
        f'{conf_line}'
        f'Время: {time.strftime("%Y-%m-%d %H:%M:%S")}'
    )
    # В письмо вкладываем уже сохранённый скриншот события (если он есть).
    attach = screenshot if (screenshot and os.path.exists(screenshot)) else None
    _send_email_async(subject, body, attach)


def handle_events(frame, detections: list, clean_frame=None):
    now  = time.time()
    h, w = frame.shape[:2]
    # Кадр для обучающей выборки — без рамок детекции и без зон,
    # чтобы сохранённые примеры были «чистыми» (см. capture_loop).
    train_frame = clean_frame if clean_frame is not None else frame

    # Тревоги по зонам
    for ev in zone_manager.process(detections, w, h, now):
        save_event(ev['event_type'], ev['obj'], frame,
                   zone=ev['zone'], save_media=ev['save_media'])

    # Пользовательские правила тревог (rules.py): класс X в зоне Y и т.п.
    fired_rules = rule_engine.evaluate(
        zone_manager.get_membership(), current_class_counts, now)
    for fr in fired_rules:
        zone_name = zone_manager.zone_name(fr['zone_id']) if fr['zone_id'] else None
        obj = {'class': fr['matched_class'], 'track_id': None, 'confidence': None}
        # save_event сам отправит письмо-тревогу (см. _notify_email).
        save_event(f"Правило: {fr['name']}", obj, frame, zone=zone_name)

    # Аномалии АЗС
    for alert in anomaly_detector.process(detections, now):
        save_event(alert['type'], alert['obj'], frame)

    # Сбор данных для дообучения (на чистом кадре — без рамок и зон)
    hard_collector.process(train_frame, detections)
    pseudo_labeler.update(train_frame, detections)

    # Глобальные тревоги (если включены)
    cooldown   = SETTINGS.get('event_cooldown_seconds', 15)
    alert_conf = SETTINGS.get('alert_confidence', 0.7)
    appear_on  = SETTINGS.get('alert_on_appearance', False)
    long_on    = SETTINGS.get('global_long_stay_alerts', False)
    if not (appear_on or long_on):
        return

    def _cooldown_ok(key: str) -> bool:
        if now - _last_global_event.get(key, 0) < cooldown:
            return False
        _last_global_event[key] = now
        return True

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


# ---------------------------------------------------------------------------
# Поток захвата и обработки кадров
# ---------------------------------------------------------------------------

def capture_loop():
    global frame_buffer, current_detections, current_class_counts
    global current_zone_stats, detection_fps, running

    frame_count  = 0
    fps_timer    = time.time()
    last_seq     = -1
    low_fps_logged = 0.0

    logger.info('Поток захвата запущен')

    while running:
        seq, frame = rtsp_manager.read_seq()
        # Источник не выбран / не подключён — сбрасываем кадр, чтобы плеер
        # показал заглушку «нет источника», а не застывший последний кадр.
        if frame is None:
            if frame_buffer is not None:
                frame_buffer = None
            current_detections = []
            current_class_counts = {}
            detection_fps = 0
            time.sleep(0.03)
            continue
        # Обрабатываем только новые кадры — иначе YOLO гоняется по одному и тому
        # же кадру многократно, что зря греет CPU и искажает счётчик FPS.
        if seq == last_seq:
            time.sleep(0.005)
            continue
        last_seq = seq

        # Чистый кадр (до отрисовки рамок детекции) — нужен для обучающей
        # выборки, чтобы примеры не содержали наложенных рамок/зон.
        clean = frame.copy()

        processed, dets, counts = detector.process_frame(frame)
        # Зоны НЕ «вжигаем» в кадр: они рисуются SVG-оверлеем на странице
        # «Монитор» (index.html). Это убирает двойную отрисовку и оставляет
        # видеопоток (и кадры дообучения) чистыми.

        current_detections   = dets
        current_class_counts = counts

        if detector.enabled:
            handle_events(processed, dets, clean)
            current_zone_stats = zone_manager.get_live()

        frame_buffer = processed

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            detection_fps = frame_count
            frame_count   = 0
            fps_timer     = now
            # Индикация падения FPS: если детекция включена и реальный FPS
            # заметно ниже FPS источника — пишем предупреждение в лог
            # (не чаще раза в 15 сек, чтобы не засорять журнал).
            src_fps = rtsp_manager.get_status().get('stream_fps') or 0
            if (detector.enabled and src_fps >= 5
                    and detection_fps < src_fps * 0.6
                    and now - low_fps_logged > 15.0):
                logger.warning(
                    'Низкий FPS обработки: %d из %.0f (источник). '
                    'Увеличьте frame_skip или снизьте разрешение.',
                    detection_fps, src_fps)
                low_fps_logged = now


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global running
    database.init_db()
    running = True

    rtsp_manager.start()
    t = threading.Thread(target=capture_loop, daemon=True, name='capture')
    t.start()
    cloud_sync.start()

    print('=' * 55)
    print('  Веб-приложение камеры АЗС')
    print('  http://localhost:8000')
    print('=' * 55)
    logger.info('Сервер запущен')
    yield

    logger.info('Сервер останавливается...')
    running = False
    rtsp_manager.stop()    # stop() гарантирует release VideoCapture
    cloud_sync.stop()
    t.join(timeout=4)
    logger.info('Сервер остановлен')


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title='Камера АЗС', lifespan=lifespan)
app.mount('/detections', StaticFiles(directory='detections'), name='detections')
app.mount('/static',     StaticFiles(directory='static'),     name='static')


# ---------------------------------------------------------------------------
# MJPEG стрим
# ---------------------------------------------------------------------------

import numpy as np

_placeholder_bytes: bytes | None = None


def _placeholder_jpeg(status: str = '') -> bytes:
    """
    Тёмный кадр-заставка (без текста — OpenCV не умеет кириллицу).
    Понятное сообщение о статусе показывает HTML-оверлей поверх плеера
    (см. index.html), поэтому здесь нужен лишь нейтральный фон.
    """
    global _placeholder_bytes
    if _placeholder_bytes is not None:
        return _placeholder_bytes
    img = np.full((360, 640, 3), (18, 14, 10), dtype=np.uint8)
    cv2.rectangle(img, (1, 1), (638, 358), (40, 56, 78), 1)
    # простая «камера выключена»: круг с диагональю по центру
    cv2.circle(img, (320, 175), 34, (70, 90, 120), 2)
    cv2.line(img, (298, 153), (342, 197), (70, 90, 120), 2)
    ok, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _placeholder_bytes = jpeg.tobytes() if ok else b''
    return _placeholder_bytes


def _mjpeg_gen():
    """
    Постоянный генератор MJPEG: одно соединение на всё время работы страницы.
    Если кадра нет (источник не выбран/подключается) — отдаёт заставку с текстом.
    Благодаря этому смена источника применяется БЕЗ перезагрузки страницы:
    то же соединение плавно переходит с заставки на реальные кадры.
    """
    prev_id = None
    last_placeholder = 0.0
    while running:
        fb = frame_buffer
        if fb is not None:
            if id(fb) != prev_id:
                prev_id = id(fb)
                ok, jpeg = cv2.imencode('.jpg', fb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + jpeg.tobytes() + b'\r\n')
            else:
                time.sleep(0.02)
            continue
        # Кадра нет — периодически отдаём заставку (держим соединение живым)
        now = time.time()
        if now - last_placeholder > 0.4:
            last_placeholder = now
            prev_id = None
            data = _placeholder_jpeg(rtsp_manager.get_status().get('status', 'no_source'))
            if data:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        else:
            time.sleep(0.05)


@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(
        _mjpeg_gen(),
        media_type='multipart/x-mixed-replace; boundary=frame',
    )


def _mjpeg_raw_gen():
    """
    «Сырой» MJPEG-поток напрямую от источника — без рамок детекции.
    Используется на странице дообучения (ручная разметка), где оператору
    нужно видеть чистое видео, чтобы отбирать кадры для разметки.
    """
    prev_seq = None
    last_placeholder = 0.0
    while running:
        seq, frame = rtsp_manager.read_seq()
        if frame is not None:
            if seq != prev_seq:
                prev_seq = seq
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + jpeg.tobytes() + b'\r\n')
            else:
                time.sleep(0.02)
            continue
        now = time.time()
        if now - last_placeholder > 0.4:
            last_placeholder = now
            prev_seq = None
            data = _placeholder_jpeg()
            if data:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        else:
            time.sleep(0.05)


@app.get('/video_feed_raw')
async def video_feed_raw():
    return StreamingResponse(
        _mjpeg_raw_gen(),
        media_type='multipart/x-mixed-replace; boundary=frame',
    )


# ---------------------------------------------------------------------------
# HTML-страницы
# ---------------------------------------------------------------------------

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

@app.get('/retrain',  response_class=HTMLResponse)
async def page_retrain():  return HTMLResponse(_read_html('retrain.html'))

@app.get('/manage',   response_class=HTMLResponse)
async def page_manage():   return HTMLResponse(_read_html('manage.html'))


# ---------------------------------------------------------------------------
# API: детекция
# ---------------------------------------------------------------------------

@app.get('/api/detections')
async def api_detections():
    return {
        'detection_enabled': detector.enabled,
        'fps':               detection_fps,
        'source_fps':        round(rtsp_manager.get_status().get('stream_fps') or 0),
        'objects':           current_detections,
        'class_counts':      current_class_counts,
        'zones':             current_zone_stats,
    }


@app.post('/api/detection/start')
async def api_start():
    # Нет источника — детектировать нечего. Возвращаем понятную причину.
    if rtsp_manager.get_status().get('status') == RTSPManager.STATUS_NO_SOURCE:
        return JSONResponse(
            {'status': 'error', 'error': 'Источник видео не выбран'}, 400)
    _alerted_appear.clear()
    _alerted_long.clear()
    _last_global_event.clear()
    _rule_fire_counts.clear()
    _email_last_sent.clear()
    rule_engine.reset_runtime()
    zone_manager.reset_runtime()
    anomaly_detector.reset()
    detector.set_enabled(True)
    return {'status': 'started'}


@app.post('/api/detection/stop')
async def api_stop():
    global current_zone_stats
    detector.set_enabled(False)
    _rule_fire_counts.clear()
    _email_last_sent.clear()
    rule_engine.reset_runtime()
    zone_manager.reset_runtime()
    anomaly_detector.reset()
    current_zone_stats = zone_manager.get_live()
    return {'status': 'stopped'}


# ---------------------------------------------------------------------------
# API: видеопоток — смена источника, загрузка файла, проверка камеры
# ---------------------------------------------------------------------------

@app.get('/api/stream/status')
async def api_stream_status():
    return rtsp_manager.get_status()


@app.post('/api/stream/change')
async def api_stream_change(request: Request):
    """
    Горячая смена источника видео.
    Body: {"source": "0"} | {"source": "rtsp://..."} | {"source": "data/uploads/vid.mp4"}
    """
    global SETTINGS
    data       = await request.json()
    new_source = (data.get('source') or '').strip()
    # Пустой source допустим — означает «отключить источник» (нет видео).

    rtsp_manager.change_source(new_source)
    SETTINGS = config.save_settings({'video_source': new_source})
    logger.info('Источник видео изменён: %s', new_source or '(нет источника)')
    return {'ok': True, 'source': new_source}


@app.post('/api/stream/disconnect')
async def api_stream_disconnect():
    """Отключает текущий источник видео (переводит плеер в режим «нет источника»)."""
    global SETTINGS
    detector.set_enabled(False)
    rtsp_manager.change_source('')
    SETTINGS = config.save_settings({'video_source': ''})
    logger.info('Источник видео отключён')
    return {'ok': True, 'source': ''}


@app.post('/api/stream/upload')
async def api_stream_upload(file: UploadFile = File(...)):
    """
    Загрузка видеофайла с клиента.
    После загрузки автоматически переключает поток на этот файл.
    Поддерживаемые форматы: mp4, avi, mov, mkv, webm.
    """
    global SETTINGS

    ALLOWED_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.ts', '.m4v'}
    suffix = Path(file.filename or 'video').suffix.lower()
    if suffix not in ALLOWED_EXT:
        return JSONResponse(
            {'ok': False, 'error': f'Недопустимый формат: {suffix}. '
                                   f'Разрешены: {", ".join(ALLOWED_EXT)}'},
            400,
        )

    # Сохраняем файл
    safe_name = f"upload_{int(time.time())}{suffix}"
    dest      = os.path.join(UPLOADS_DIR, safe_name)
    try:
        with open(dest, 'wb') as f_out:
            shutil.copyfileobj(file.file, f_out)
    except Exception as exc:
        logger.error('Ошибка сохранения загруженного файла: %s', exc)
        return JSONResponse({'ok': False, 'error': str(exc)}, 500)
    finally:
        await file.close()

    size_mb = os.path.getsize(dest) / 1024 / 1024
    logger.info('Файл загружен: %s (%.1f МБ)', dest, size_mb)

    # Переключаем поток
    rtsp_manager.change_source(dest)
    SETTINGS = config.save_settings({'video_source': dest})

    return {'ok': True, 'source': dest, 'filename': safe_name, 'size_mb': round(size_mb, 2)}


@app.post('/api/stream/probe')
async def api_stream_probe(request: Request):
    """Проверяет доступность источника видео (не переключает)."""
    data = await request.json()
    url  = data.get('url', '').strip()
    if not url:
        return JSONResponse({'ok': False, 'error': 'url не задан'}, 400)
    # probe_source блокирует — запускаем в потоке чтобы не блокировать event loop
    import asyncio
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, probe_source, url, 5.0)
    return result


# ---------------------------------------------------------------------------
# API: события
# ---------------------------------------------------------------------------

@app.get('/api/events')
async def api_events(
    page: int = 1, per_page: int = 10,
    event_type: str = '', date_from: str = '',
    date_to: str = '', zone: str = '',
):
    et = event_type or None
    df = date_from  or None
    dt = date_to    or None
    zn = zone       or None
    total = database.count_events(et, df, dt, zn)
    items = database.get_events(
        limit=per_page, offset=(page - 1) * per_page,
        event_type=et, date_from=df, date_to=dt, zone=zn,
    )
    return {'events': items, 'total': total, 'page': page, 'per_page': per_page}


@app.get('/api/events/zone_report')
async def api_zone_report(date_from: str = '', date_to: str = ''):
    return {'report': database.zone_report(date_from or None, date_to or None)}


@app.post('/api/events/clear')
async def api_events_clear():
    database.clear_events()
    return {'status': 'cleared'}


# ---------------------------------------------------------------------------
# API: зоны
# ---------------------------------------------------------------------------

@app.get('/api/zones')
async def api_get_zones():
    return {'zones': zone_store.load_all()}


@app.post('/api/zones')
async def api_save_zones(request: Request):
    data  = await request.json()
    zones = data.get('zones', [])
    saved = zone_store.save_all(zones)
    zone_manager.update_zones(saved)
    return {'status': 'saved', 'zones': saved}


@app.get('/api/zones/export')
async def api_export_zone(id: str):
    """Экспорт одной зоны в JSON-файл (zoneX.json)."""
    zone = zone_store.export_zone(id)
    if zone is None:
        return JSONResponse({'ok': False, 'error': 'Зона не найдена'}, 404)
    # Имя зоны может быть на кириллице, а заголовок Content-Disposition
    # допускает только latin-1. Используем ASCII-безопасное имя файла по id.
    safe = ''.join(ch for ch in str(zone.get('id', 'zone')) if ch.isalnum() or ch in '-_')
    return JSONResponse(
        zone,
        headers={'Content-Disposition': f'attachment; filename="zone-{safe or "export"}.json"'})


@app.post('/api/zones/import')
async def api_import_zone(request: Request):
    """Импорт зоны из JSON. Зона добавляется выключенной."""
    data   = await request.json()
    result = zone_store.import_zone(data)
    if result.get('ok'):
        zone_manager.update_zones(zone_store.load_all())
    return result


# ---------------------------------------------------------------------------
# API: классы объектов (создание пользовательских классов)
# ---------------------------------------------------------------------------

@app.get('/api/classes')
async def api_get_classes():
    return {'classes': class_registry.list_classes()}


@app.post('/api/classes')
async def api_add_class(request: Request):
    data   = await request.json()
    result = class_registry.add_class(data.get('name', ''), data.get('color', '#a78bfa'))
    if result.get('ok'):
        detector.refresh_classes()
    return result


@app.put('/api/classes/{class_id}')
async def api_update_class(class_id: int, request: Request):
    data   = await request.json()
    result = class_registry.update_class(class_id, data.get('name'), data.get('color'))
    if result.get('ok'):
        detector.refresh_classes()
    return result


@app.delete('/api/classes/{class_id}')
async def api_delete_class(class_id: int):
    result = class_registry.delete_class(class_id)
    if result.get('ok'):
        detector.refresh_classes()
    return result


# ---------------------------------------------------------------------------
# API: правила тревог
# ---------------------------------------------------------------------------

@app.get('/api/rules')
async def api_get_rules():
    return {'rules': rule_engine.list_rules()}


@app.post('/api/rules')
async def api_save_rule(request: Request):
    data = await request.json()
    return rule_engine.save_rule(data)


@app.delete('/api/rules/{rule_id}')
async def api_delete_rule(rule_id: str):
    return rule_engine.delete_rule(rule_id)


# ---------------------------------------------------------------------------
# API: диспетчер тревог (email)
# ---------------------------------------------------------------------------

@app.post('/api/alerts/test')
async def api_alerts_test():
    """Тестовое письмо по текущим настройкам SMTP."""
    cfg = {k: SETTINGS.get(k) for k in (
        'email_enabled', 'email_smtp_host', 'email_smtp_port', 'email_use_tls',
        'email_login', 'email_password', 'email_from', 'email_to')}
    # для теста временно считаем отправку включённой
    cfg['email_enabled'] = True
    ok, msg = email_notify.test_connection(cfg)
    return JSONResponse({'ok': ok, 'message': msg})


# ---------------------------------------------------------------------------
# API: ручная разметка кадров (без автоматической детекции)
# ---------------------------------------------------------------------------

@app.get('/api/retrain/grab_frame')
async def api_grab_frame():
    """
    Возвращает текущий «чистый» кадр видеопотока (без рамок детекции) в JPEG.
    Работает независимо от того, включена детекция или нет — оператор может
    отбирать кадры вручную.
    """
    frame = rtsp_manager.read()
    if frame is None:
        return JSONResponse({'error': 'Нет кадра от источника'}, 503)
    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return JSONResponse({'error': 'Ошибка кодирования кадра'}, 500)
    from fastapi.responses import Response
    h, w = frame.shape[:2]
    return Response(content=jpeg.tobytes(), media_type='image/jpeg',
                    headers={'X-Frame-Width': str(w), 'X-Frame-Height': str(h)})


@app.post('/api/retrain/manual_save')
async def api_manual_save(request: Request):
    """
    Сохраняет вручную размеченный кадр в обучающую выборку.
    Тело: {bbox:[x1,y1,x2,y2] (пиксели исходного кадра), class:"имя"}.
    Кадр берётся текущий из видеопотока (без детекции).
    """
    data  = await request.json()
    bbox  = data.get('bbox')
    cls   = (data.get('class') or '').strip()
    frame = rtsp_manager.read()
    if frame is None:
        return JSONResponse({'ok': False, 'error': 'Нет кадра от источника'}, 503)
    return trainer.add_manual_example(frame, bbox, cls)


# ---------------------------------------------------------------------------
# API: настройки
# ---------------------------------------------------------------------------

@app.get('/api/settings')
async def api_get_settings():
    return SETTINGS


@app.post('/api/settings')
async def api_save_settings(request: Request):
    global SETTINGS
    data     = await request.json()
    SETTINGS = config.save_settings(data)

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
    # Зоны хранятся в zone_store (zones/zoneX.json), а не в settings.json —
    # сохранение прочих настроек не должно затирать активные зоны.
    zone_manager.update_zones(zone_store.load_all())
    zone_manager.cooldown_default = SETTINGS.get('event_cooldown_seconds', 15)

    # Смена источника через настройки (если поле изменилось).
    # Пустое значение допустимо — означает отключение источника.
    new_src = SETTINGS.get('video_source', '')
    if new_src != rtsp_manager.get_status()['source']:
        rtsp_manager.change_source(new_src)

    return {'status': 'saved', 'settings': SETTINGS}


# ---------------------------------------------------------------------------
# API: дообучение
# ---------------------------------------------------------------------------

@app.get('/api/retrain/status')
async def api_retrain_status():
    trainer_status  = trainer.get_status()
    collector_stats = hard_collector.get_stats()
    labeler_stats   = pseudo_labeler.get_stats()
    return {
        **trainer_status,
        **collector_stats,
        **labeler_stats,
        'hard_count':      collector_stats.get('hard_examples_total', 0),
        'pending_count':   labeler_stats.get('pending_total', 0),
        'confirmed_count': labeler_stats.get('confirmed_total', 0),
    }


@app.get('/api/retrain/pending')
async def api_retrain_pending():
    items = pseudo_labeler.get_pending()
    return {'items': items, 'total': len(items)}


@app.get('/api/retrain/hard_examples')
async def api_retrain_hard():
    from core.incremental_trainer import _load_meta
    items = []
    for img_path in sorted(DIR_HARD.glob('*.jpg'), reverse=True)[:100]:
        meta = _load_meta(img_path)
        items.append({
            'filename':        img_path.name,
            'suggested_class': meta.get('suggested_class', 'unknown'),
            'confidence':      meta.get('confidence', 0.0),
            'track_id':        meta.get('track_id', -1),
            'reason':          meta.get('reason', ''),
            'timestamp':       meta.get('timestamp', ''),
        })
    return {'items': items, 'total': len(items)}


@app.get('/api/retrain/image')
async def api_retrain_image(file: str, dir: str = 'pending'):
    base      = DIR_PENDING if dir == 'pending' else DIR_HARD
    safe_name = Path(file).name
    img_path  = base / safe_name
    if not img_path.exists() or img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        return JSONResponse({'error': 'Файл не найден'}, 404)
    return FileResponse(str(img_path), media_type='image/jpeg')


@app.post('/api/retrain/confirm')
async def api_retrain_confirm(request: Request):
    data            = await request.json()
    filename        = data.get('filename', '')
    confirmed_class = data.get('confirmed_class', '')
    if not filename or not confirmed_class:
        return JSONResponse({'ok': False, 'error': 'filename и confirmed_class обязательны'}, 400)
    return {'ok': trainer.confirm_label(filename, confirmed_class)}


@app.post('/api/retrain/reject')
async def api_retrain_reject(request: Request):
    data     = await request.json()
    filename = data.get('filename', '')
    if not filename:
        return JSONResponse({'ok': False, 'error': 'filename обязателен'}, 400)
    return {'ok': trainer.reject_label(filename)}


@app.post('/api/retrain/confirm_all')
async def api_retrain_confirm_all():
    count = trainer.confirm_all()
    return {'ok': True, 'confirmed': count}


@app.post('/api/retrain/clear_pending')
async def api_retrain_clear_pending():
    count = trainer.clear_pending()
    return {'ok': True, 'cleared': count}


@app.post('/api/retrain/hard_confirm')
async def api_retrain_hard_confirm(request: Request):
    data            = await request.json()
    filename        = data.get('filename', '')
    confirmed_class = data.get('confirmed_class', '')
    if not filename or not confirmed_class:
        return JSONResponse({'ok': False, 'error': 'filename и confirmed_class обязательны'}, 400)
    return {'ok': trainer.confirm_hard_example(filename, confirmed_class)}


@app.post('/api/retrain/hard_reject')
async def api_retrain_hard_reject(request: Request):
    data     = await request.json()
    filename = data.get('filename', '')
    if not filename:
        return JSONResponse({'ok': False, 'error': 'filename обязателен'}, 400)
    return {'ok': trainer.reject_hard_example(filename)}


@app.post('/api/retrain/clear_hard')
async def api_retrain_clear_hard():
    count = trainer.clear_hard()
    return {'ok': True, 'cleared': count}


@app.post('/api/retrain/start')
async def api_retrain_start():
    confirmed = trainer.get_confirmed_count()
    if confirmed < 30:
        return {'started': False,
                'reason': f'Недостаточно подтверждённых примеров: {confirmed} < 30'}

    def _on_model_ready(path: str):
        try:
            from ultralytics import YOLO
            detector.model = YOLO(path)
            logger.info('Горячая замена модели: %s', path)
        except Exception as exc:
            logger.error('Ошибка горячей замены модели: %s', exc)

    started = trainer.start_training(on_ready=_on_model_ready)
    return {'started': started,
            'reason': '' if started else 'Дообучение уже запущено'}


@app.get('/api/retrain/versions')
async def api_retrain_versions():
    return {'versions': trainer.get_model_versions()}


@app.post('/api/retrain/apply_version')
async def api_retrain_apply_version(request: Request):
    data     = await request.json()
    filename = data.get('filename', '')
    if not filename:
        return JSONResponse({'ok': False, 'error': 'filename обязателен'}, 400)
    path = trainer.apply_version(filename)
    if path is None:
        return JSONResponse({'ok': False, 'error': 'Версия не найдена'}, 404)
    try:
        from ultralytics import YOLO
        detector.model = YOLO(path)
        logger.info('Откат к версии: %s', filename)
    except Exception as exc:
        return JSONResponse({'ok': False, 'error': str(exc)}, 500)
    return {'ok': True, 'path': path}


# ---------------------------------------------------------------------------
# API: health / metrics
# ---------------------------------------------------------------------------

@app.get('/health')
async def health():
    return {
        'status':    'ok',
        'camera':    rtsp_manager.get_status()['status'],
        'detection': detector.enabled,
        'db':        os.path.exists('data/events.db'),
        'model':     os.path.exists(_best_model_path()),
    }


@app.get('/metrics')
async def metrics():
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
    except ImportError:
        cpu, mem = 0.0, 0.0
    return {
        'fps':         detection_fps,
        'cpu_percent': cpu,
        'mem_percent': mem,
        'alerts_db':   database.count_events(),
        'stream':      rtsp_manager.get_status()['status'],
    }


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
