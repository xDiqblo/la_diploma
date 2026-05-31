"""
video_buffer.py — кольцевой буфер видео для сохранения фрагментов при тревоге.

Идея: постоянно храним в памяти последние N секунд кадров (буфер «ДО»).
При тревоге фиксируем эти кадры и продолжаем записывать ещё M секунд («ПОСЛЕ»),
после чего склеиваем всё в один MP4-файл. Запись файла идёт в отдельном потоке,
чтобы не тормозить захват видео.
"""
import os
import time
import threading
from collections import deque

import cv2


class VideoClipBuffer:
    def __init__(self, fps=25, pre_seconds=10, post_seconds=10, out_dir='detections'):
        self.fps = max(1, int(fps))
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Буфер «ДО»: храним последние pre_seconds * fps кадров
        self.buffer = deque(maxlen=self.fps * pre_seconds)
        self.lock = threading.Lock()

        # Состояние записи «ПОСЛЕ»
        self.recording = False
        self.post_frames = []
        self.post_target = 0
        self.pending_path = None
        self.on_complete = None  # callback(path) — вызовется, когда клип готов

    def add_frame(self, frame):
        """Вызывается на КАЖДОМ кадре из основного цикла захвата."""
        with self.lock:
            self.buffer.append(frame.copy())
            if self.recording:
                self.post_frames.append(frame.copy())
                if len(self.post_frames) >= self.post_target:
                    # Набрали достаточно кадров «ПОСЛЕ» — сохраняем клип
                    pre_frames = list(self.buffer)
                    post_frames = self.post_frames
                    path = self.pending_path
                    callback = self.on_complete
                    # Сбрасываем состояние записи
                    self.recording = False
                    self.post_frames = []
                    self.pending_path = None
                    # Пишем файл в отдельном потоке
                    threading.Thread(
                        target=self._write_clip,
                        args=(pre_frames + post_frames, path, callback),
                        daemon=True,
                    ).start()

    def start_clip(self, on_complete=None):
        """
        Запускает сохранение фрагмента: буфер «ДО» + запись «ПОСЛЕ».
        Возвращает путь к будущему файлу (или None, если запись уже идёт).
        on_complete(path) — необязательный колбэк по готовности файла.
        """
        with self.lock:
            if self.recording:
                return None  # уже записываем один клип — второй не начинаем
            filename = f"clip_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            path = os.path.join(self.out_dir, filename)
            self.recording = True
            self.post_frames = []
            self.post_target = self.fps * self.post_seconds
            self.pending_path = path
            self.on_complete = on_complete
            return path

    def _write_clip(self, frames, path, callback):
        """Записывает список кадров в MP4-файл."""
        if not frames:
            return
        try:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            print(f"[VideoBuffer] Клип сохранён: {path} ({len(frames)} кадров)")
            if callback:
                callback(path)
        except Exception as e:
            print(f"[VideoBuffer] Ошибка записи клипа: {e}")
