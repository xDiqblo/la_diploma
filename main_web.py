import cv2
import time
import threading
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import Detector
import config
import database
import telegram_notify
from video_buffer import VideoClipBuffer

# Папки для шаблонов и сохранённых тревог
os.makedirs('templates', exist_ok=True)
os.makedirs('detections', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ========== Глобальные переменные ==========
frame_buffer = None             # последний обработанный кадр (для MJPEG)
current_detections = []         # текущие объекты в кадре
current_class_counts = {}       # счётчики по классам
detection_fps = 0               # FPS обработки
capture_thread = None
running = True
video_clip_buffer = None        # кольцевой буфер видео (создаётся в потоке захвата)

# Множества уже обработанных track_id (чтобы не плодить дубли событий)
alerted_appearance = set()
alerted_long_stay = set()

# Загружаем настройки из JSON
SETTINGS = config.load_settings()

# Инициализация детектора с параметрами из настроек
detector = Detector(
    model_path='yolov8n.pt',
    confidence_threshold=SETTINGS['confidence_threshold'],
    frame_skip=SETTINGS['frame_skip'],
    treat_person_as_car=SETTINGS['treat_person_as_car'],
    long_stay_seconds=SETTINGS['long_stay_seconds'],
)

# Источник видео (файл, RTSP-ссылка или индекс камеры, например 0)
VIDEO_SOURCE = 'test_video3.mp4'


# ========== Сохранение события (тревоги) ==========
def save_event(event_type, obj, frame):
    """Сохраняет скриншот, запись в БД, видеофрагмент и шлёт Telegram-уведомление."""
    class_name = obj['class']
    track_id = obj['track_id']
    confidence = obj['confidence']

    # 1. Скриншот
    screenshot_path = None
    if SETTINGS.get('save_screenshots', True):
        filename = f"shot_{time.strftime('%Y%m%d_%H%M%S')}_id{track_id}.jpg"
        full_path = os.path.join('detections', filename)
        cv2.imwrite(full_path, frame)
        screenshot_path = f"detections/{filename}"  # относительный путь для веба

    # 2. Запись в БД
    event_id = database.add_event(
        event_type=event_type,
        class_name=class_name,
        track_id=track_id,
        confidence=confidence,
        screenshot=screenshot_path,
    )

    # 3. Видеофрагмент (10 сек до + 10 сек после) — асинхронно
    if SETTINGS.get('save_video_clips', True) and video_clip_buffer is not None:
        def _on_clip_ready(path):
            rel = path.replace('\\', '/')
            database.update_event_clip(event_id, rel)
        video_clip_buffer.start_clip(on_complete=_on_clip_ready)

    # 4. Telegram-уведомление
    if SETTINGS.get('telegram_enabled', False):
        token = SETTINGS.get('telegram_token', '')
        chat_id = SETTINGS.get('telegram_chat_id', '')
        caption = (f"🚨 {event_type}\n"
                   f"Объект: {class_name} (ID {track_id})\n"
                   f"Уверенность: {confidence}\n"
                   f"Время: {time.strftime('%H:%M:%S')}")
        if screenshot_path and os.path.exists(screenshot_path):
            telegram_notify.send_photo(token, chat_id, screenshot_path, caption)
        else:
            telegram_notify.send_message(token, chat_id, caption)

    print(f"[Событие] {event_type}: {class_name} ID{track_id} conf={confidence}")


def handle_events(frame, detections):
    """Анализирует детекции и создаёт события при тревогах."""
    alert_conf = SETTINGS.get('alert_confidence', 0.7)
    for obj in detections:
        track_id = obj['track_id']
        if track_id is None:
            continue

        # Событие 1: появление уверенно распознанного объекта
        if track_id not in alerted_appearance and obj['confidence'] >= alert_conf:
            alerted_appearance.add(track_id)
            save_event('Появление объекта', obj, frame)

        # Событие 2: объект слишком долго в кадре (аномалия)
        if obj.get('long_stay') and track_id not in alerted_long_stay:
            alerted_long_stay.add(track_id)
            save_event('Долгое нахождение', obj, frame)


# ========== Функция захвата и обработки видео ==========
def capture_and_detect():
    global frame_buffer, current_detections, current_class_counts
    global detection_fps, running, video_clip_buffer

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть источник видео {VIDEO_SOURCE}")
        return

    # Реальный FPS видео (для контроля скорости воспроизведения файла)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25
    frame_delay = 1.0 / video_fps
    print(f"FPS видео: {video_fps}, задержка: {frame_delay:.3f} сек")

    # Создаём кольцевой буфер видео под реальный FPS
    video_clip_buffer = VideoClipBuffer(
        fps=int(video_fps),
        pre_seconds=SETTINGS.get('clip_pre_seconds', 10),
        post_seconds=SETTINGS.get('clip_post_seconds', 10),
        out_dir='detections',
    )

    frame_count = 0
    fps_start_time = time.time()
    last_frame_time = time.time()

    while running:
        # Контроль скорости воспроизведения (для видеофайла)
        now = time.time()
        time_since_last = now - last_frame_time
        if time_since_last < frame_delay:
            time.sleep(frame_delay - time_since_last)

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # зацикливаем файл
            continue

        last_frame_time = time.time()

        # Обработка кадра детектором
        processed_frame, detections, class_counts = detector.process_frame(frame)

        frame_buffer = processed_frame
        current_detections = detections
        current_class_counts = class_counts

        # Когда детекция включена — кормим видеобуфер и проверяем тревоги
        if detector.enabled:
            video_clip_buffer.add_frame(processed_frame)
            handle_events(processed_frame, detections)

        # Подсчёт FPS обработки
        frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            detection_fps = frame_count
            frame_count = 0
            fps_start_time = time.time()

    cap.release()


# ========== Lifespan менеджер ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global capture_thread, running
    database.init_db()  # создаём таблицу событий
    running = True
    capture_thread = threading.Thread(target=capture_and_detect, daemon=True)
    capture_thread.start()
    print("✅ Фоновый поток захвата видео запущен")
    yield
    running = False
    if capture_thread:
        capture_thread.join(timeout=2)
    print("✅ Фоновый поток остановлен")


# ========== Создаём приложение ==========
web_app = FastAPI(title="Камера АЗС с детекцией", lifespan=lifespan)

# Отдаём сохранённые скриншоты и видеофрагменты по URL /detections/...
web_app.mount("/detections", StaticFiles(directory="detections"), name="detections")
web_app.mount("/static", StaticFiles(directory="static"), name="static")


# ========== Генератор MJPEG-потока ==========
def generate_mjpeg_stream():
    while running:
        if frame_buffer is not None:
            _, jpeg = cv2.imencode('.jpg', frame_buffer, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() +
                   b'\r\n')
        else:
            time.sleep(0.01)


# ========== Чтение HTML из файла ==========
def get_html(name):
    html_path = os.path.join('templates', name)
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"<h1>Файл templates/{name} не найден</h1>"


# ========== HTML-страницы ==========
@web_app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=get_html('index.html'))


@web_app.get("/events", response_class=HTMLResponse)
async def events_page():
    return HTMLResponse(content=get_html('events.html'))


@web_app.get("/settings", response_class=HTMLResponse)
async def settings_page():
    return HTMLResponse(content=get_html('settings.html'))


# ========== Видеопоток ==========
@web_app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_mjpeg_stream(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ========== API: детекции ==========
@web_app.get("/api/detections")
async def get_detections():
    return {
        "detection_enabled": detector.enabled,
        "fps": detection_fps,
        "objects": current_detections,
        "class_counts": current_class_counts,
    }


@web_app.post("/api/detection/start")
async def start_detection():
    # При запуске очищаем историю тревог, чтобы события начинались «с чистого листа»
    alerted_appearance.clear()
    alerted_long_stay.clear()
    detector.set_enabled(True)
    return {"status": "started"}


@web_app.post("/api/detection/stop")
async def stop_detection():
    detector.set_enabled(False)
    return {"status": "stopped"}


# ========== API: события ==========
@web_app.get("/api/events")
async def api_events():
    return {"events": database.get_events(limit=200)}


@web_app.post("/api/events/clear")
async def api_events_clear():
    database.clear_events()
    return {"status": "cleared"}


# ========== API: настройки ==========
@web_app.get("/api/settings")
async def api_get_settings():
    return SETTINGS


@web_app.post("/api/settings")
async def api_save_settings(request: Request):
    global SETTINGS
    data = await request.json()
    # Сохраняем в файл и обновляем глобальные настройки
    SETTINGS = config.save_settings(data)
    # Применяем к детектору «на лету»
    detector.update_settings(
        confidence_threshold=SETTINGS['confidence_threshold'],
        frame_skip=SETTINGS['frame_skip'],
        treat_person_as_car=SETTINGS['treat_person_as_car'],
        long_stay_seconds=SETTINGS['long_stay_seconds'],
    )
    return {"status": "saved", "settings": SETTINGS}


@web_app.post("/api/telegram/test")
async def api_telegram_test():
    ok, msg = telegram_notify.test_connection(
        SETTINGS.get('telegram_token', ''),
        SETTINGS.get('telegram_chat_id', ''),
    )
    return JSONResponse({"ok": ok, "message": msg})


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("Запуск веб-приложения камеры АЗС")
    print("Откройте в браузере: http://localhost:8000")
    print("=" * 50)
    uvicorn.run(web_app, host="0.0.0.0", port=8000)
