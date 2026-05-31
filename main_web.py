import cv2
import time
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import os

os.makedirs('templates', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

from detector import Detector

# ========== Глобальные переменные ==========
frame_buffer = None
current_detections = []
current_class_counts = {}
detection_fps = 0
capture_thread = None
running = True

# Инициализация детектора (frame_skip=2 — обрабатываем каждый 2-й кадр)
detector = Detector(model_path='yolov8n.pt', confidence_threshold=0.5, frame_skip=2)

# Источник видео
VIDEO_SOURCE = 'test_video3.mp4'  # или 'test.mp4'

# Контроль FPS видео
video_fps = 25  # по умолчанию
last_frame_time = 0


# ========== Функция захвата и обработки видео ==========
def capture_and_detect():
    global frame_buffer, current_detections, current_class_counts, detection_fps, running, last_frame_time, video_fps

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть источник видео {VIDEO_SOURCE}")
        return

    # Получаем реальный FPS видео (для файлов)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25
    frame_delay = 1.0 / video_fps

    print(f"FPS видео: {video_fps}, задержка: {frame_delay:.3f} сек")

    frame_count = 0
    fps_start_time = time.time()
    last_frame_time = time.time()

    while running:
        # Контроль скорости воспроизведения
        current_time = time.time()
        time_since_last = current_time - last_frame_time
        if time_since_last < frame_delay:
            time.sleep(max(0, frame_delay - time_since_last))

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        last_frame_time = time.time()

        # Обрабатываем кадр
        processed_frame, detections, class_counts = detector.process_frame(frame)

        frame_buffer = processed_frame
        current_detections = detections
        current_class_counts = class_counts

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
    global capture_thread
    capture_thread = threading.Thread(target=capture_and_detect, daemon=True)
    capture_thread.start()
    print("✅ Фоновый поток захвата видео запущен")
    yield
    global running
    running = False
    if capture_thread:
        capture_thread.join(timeout=2)
    print("✅ Фоновый поток остановлен")


# ========== Создаём приложение ==========
web_app = FastAPI(title="Камера АЗС с детекцией", lifespan=lifespan)


# ========== Генератор MJPEG-потока ==========
def generate_mjpeg_stream():
    global frame_buffer
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
def get_html():
    html_path = os.path.join('templates', 'index.html')
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Файл templates/index.html не найден</h1>"


# ========== Эндпоинты ==========
@web_app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=get_html())


@web_app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@web_app.get("/api/detections")
async def get_detections():
    return {
        "detection_enabled": detector.enabled,
        "fps": detection_fps,
        "objects": current_detections,
        "class_counts": current_class_counts
    }


@web_app.post("/api/detection/start")
async def start_detection():
    detector.set_enabled(True)
    return {"status": "started"}


@web_app.post("/api/detection/stop")
async def stop_detection():
    detector.set_enabled(False)
    return {"status": "stopped"}


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("Запуск веб-приложения камеры АЗС")
    print("Откройте в браузере: http://localhost:8000")
    print("=" * 50)
    uvicorn.run(web_app, host="0.0.0.0", port=8000)