"""
video_buffer.py — кольцевой буфер видео для сохранения клипов при тревоге.

Принцип:
  - Буфер хранит последние (pre_seconds * fps) кадров — это запись «ДО».
  - При тревоге: фиксируем буфер «ДО» и дописываем ещё post_seconds секунд «ПОСЛЕ».
  - Склеиваем в MP4 в фоновом потоке — не блокируем захват видео.
"""
import os
import time
import threading
from collections import deque

import cv2


class VideoClipBuffer:
    def __init__(self, fps: int = 25, pre_seconds: int = 10,
                 post_seconds: int = 10, out_dir: str = 'detections'):
        self.fps = max(1, int(fps))
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Кольцевой буфер кадров «ДО»
        self._buf: deque = deque(maxlen=self.fps * pre_seconds)
        self._lock = threading.Lock()

        # Состояние записи «ПОСЛЕ»
        self._recording = False
        self._post_frames: list = []
        self._post_target: int = 0
        self._pending_path: str = ''
        self._on_complete = None  # callback(path: str)

    def add_frame(self, frame):
        """Вызывается на каждом кадре из потока захвата."""
        with self._lock:
            self._buf.append(frame.copy())
            if self._recording:
                self._post_frames.append(frame.copy())
                if len(self._post_frames) >= self._post_target:
                    # Накопили кадры «ПОСЛЕ» — запускаем запись файла
                    all_frames = list(self._buf) + self._post_frames
                    path = self._pending_path
                    cb = self._on_complete
                    self._recording = False
                    self._post_frames = []
                    threading.Thread(
                        target=self._write_mp4,
                        args=(all_frames, path, cb),
                        daemon=True,
                    ).start()

    def start_clip(self, on_complete=None) -> str | None:
        """
        Запускает сохранение клипа.
        Возвращает путь к будущему файлу или None, если запись уже идёт.
        on_complete(path) — вызывается когда MP4 готов.
        """
        with self._lock:
            if self._recording:
                return None  # уже пишем — второй клип не начинаем
            fname = f"clip_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            path = os.path.join(self.out_dir, fname)
            self._recording = True
            self._post_frames = []
            self._post_target = self.fps * self.post_seconds
            self._pending_path = path
            self._on_complete = on_complete
            return path

    @staticmethod
    def _write_mp4(frames: list, path: str, callback):
        """Пишет кадры в MP4. Вызывается в отдельном потоке."""
        if not frames:
            return
        try:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(path, fourcc, 25, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            print(f'[VideoBuffer] Клип готов: {path} ({len(frames)} кадров)')
            if callback:
                callback(path)
        except Exception as e:
            print(f'[VideoBuffer] Ошибка записи: {e}')
