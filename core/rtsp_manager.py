"""
rtsp_manager.py — управление видеопотоком с автоматическим восстановлением.

Ключевые изменения:
  - Пустой источник ('') означает «ничего не подключать» — стрим ждёт.
  - change_source() прерывает cap.read() немедленно через release().
  - stop() ждёт завершения потока с таймаутом.
  - Видеофайл зацикливается только если источник не менялся.
  - read_seq() позволяет потребителю обрабатывать только новые кадры.
"""

import logging
import threading
import time
from typing import Optional

import cv2

logger = logging.getLogger(__name__)

MAX_RECONNECT_DELAY = 30
FIRST_FRAME_TIMEOUT = 8  # сек


class RTSPManager:
    STATUS_IDLE         = 'idle'
    STATUS_CONNECTING   = 'connecting'
    STATUS_CONNECTED    = 'connected'
    STATUS_RECONNECTING = 'reconnecting'
    STATUS_STOPPED      = 'stopped'

    def __init__(self, source: str = ''):
        self._source_str = source

        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock   = threading.Lock()

        self._last_frame = None
        self._frame_seq  = 0
        self._frame_lock = threading.Lock()

        self._status         = self.STATUS_STOPPED
        self._error_msg      = ''
        self._reconnect_attempts = 0
        self._connected_at: Optional[float] = None
        self.stream_fps: float = 25.0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._wakeup = threading.Event()

        self._change_requested = False
        self._source_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._wakeup.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='rtsp-capture'
        )
        self._thread.start()
        logger.info('RTSPManager: запущен [%s]', self._source_str or '<нет источника>')

    def stop(self):
        self._running = False
        self._wakeup.set()
        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                logger.warning('RTSPManager: поток не завершился за 6 сек')
        self._status = self.STATUS_STOPPED
        logger.info('RTSPManager: остановлен')

    def read(self) -> Optional[any]:
        with self._frame_lock:
            return self._last_frame.copy() if self._last_frame is not None else None

    def read_seq(self):
        """Возвращает (seq, frame) — для пропуска дублирующихся кадров."""
        with self._frame_lock:
            if self._last_frame is None:
                return self._frame_seq, None
            return self._frame_seq, self._last_frame.copy()

    def change_source(self, new_source: str):
        """Горячая замена источника. '' — остановить поток без переподключения."""
        with self._source_lock:
            if self._source_str == new_source:
                return
            self._source_str       = new_source
            self._change_requested = True

        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

        # Сбрасываем последний кадр при смене источника
        with self._frame_lock:
            self._last_frame = None
            self._frame_seq  = 0

        self._wakeup.set()
        logger.info('RTSPManager: источник изменён -> %s', new_source or '<нет>')

    def get_status(self) -> dict:
        uptime = None
        if self._connected_at and self._status == self.STATUS_CONNECTED:
            uptime = round(time.time() - self._connected_at)
        return {
            'status':             self._status,
            'source':             self._source_str,
            'reconnect_attempts': self._reconnect_attempts,
            'uptime_seconds':     uptime,
            'stream_fps':         self.stream_fps,
            'error':              self._error_msg,
        }

    def is_connected(self) -> bool:
        return self._status == self.STATUS_CONNECTED

    def has_source(self) -> bool:
        return bool(self._source_str and self._source_str.strip())

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------

    def _loop(self):
        reconnect_delay = 1.0

        while self._running:
            self._wakeup.clear()

            with self._source_lock:
                self._change_requested = False
                source_str = self._source_str

            # Нет источника — ждём в состоянии idle
            if not source_str or not source_str.strip():
                self._status = self.STATUS_IDLE
                self._error_msg = ''
                self._reconnect_attempts = 0
                reconnect_delay = 1.0
                self._wakeup.wait(timeout=1.0)
                continue

            source = _parse_source(source_str)
            self._status = self.STATUS_CONNECTING
            logger.info('RTSPManager: подключение к [%s]', source_str)

            cap = self._try_open(source)

            if cap is None:
                self._reconnect_attempts += 1
                self._status    = self.STATUS_RECONNECTING
                self._error_msg = f'Нет сигнала: {source_str}'
                logger.warning('RTSPManager: попытка %d провалена, пауза %.0f с',
                               self._reconnect_attempts, reconnect_delay)
                self._wakeup.wait(timeout=reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)
                continue

            with self._cap_lock:
                if not self._running or self._change_requested:
                    cap.release()
                    continue
                self._cap = cap

            raw_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            self.stream_fps     = raw_fps if 1.0 <= raw_fps <= 120.0 else 25.0
            self._status        = self.STATUS_CONNECTED
            self._connected_at  = time.time()
            self._error_msg     = ''
            reconnect_delay     = 1.0
            self._reconnect_attempts = 0
            is_file = _is_file(source_str)

            frame_interval = 1.0 / self.stream_fps if (is_file and self.stream_fps > 0) else 0.0
            next_frame_t   = time.time()
            slow_logged    = 0.0

            logger.info('RTSPManager: подключено FPS=%.1f [%s]', self.stream_fps, source_str)

            consecutive_failures = 0
            while self._running and not self._change_requested:
                ret, frame = cap.read()

                if not ret:
                    if is_file:
                        if not self._change_requested:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            next_frame_t = time.time()
                            consecutive_failures = 0
                        continue
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= 5:
                            self._status    = self.STATUS_RECONNECTING
                            self._error_msg = 'Поток прерван, переподключение...'
                            logger.warning('RTSPManager: 5 подряд ошибок, переподключение')
                            break
                        time.sleep(0.05)
                        continue

                consecutive_failures = 0
                with self._frame_lock:
                    self._last_frame = frame
                    self._frame_seq += 1

                if frame_interval > 0.0:
                    next_frame_t += frame_interval
                    delay = next_frame_t - time.time()
                    if delay > 0:
                        self._wakeup.wait(timeout=delay)
                    elif delay < -1.0:
                        now = time.time()
                        if now - slow_logged > 10.0:
                            logger.warning('RTSPManager: обработка отстаёт от FPS файла')
                            slow_logged = now
                        next_frame_t = now

            with self._cap_lock:
                if self._cap is cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    self._cap = None

        self._status = self.STATUS_STOPPED

    def _try_open(self, source) -> Optional[cv2.VideoCapture]:
        try:
            if isinstance(source, str) and source.lower().startswith('rtsp://'):
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                cap.release()
                return None

            t0 = time.time()
            while time.time() - t0 < FIRST_FRAME_TIMEOUT:
                if not self._running or self._change_requested:
                    cap.release()
                    return None
                ret, _ = cap.read()
                if ret:
                    if _is_file(self._source_str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return cap
                time.sleep(0.05)

            cap.release()
            return None
        except Exception as exc:
            logger.error('RTSPManager: ошибка открытия: %s', exc)
            return None


# ------------------------------------------------------------------
# Утилиты
# ------------------------------------------------------------------

def _parse_source(src: str):
    try:
        return int(src)
    except (ValueError, TypeError):
        return src


def _is_file(src: str) -> bool:
    if not src:
        return False
    try:
        int(src)
        return False
    except (ValueError, TypeError):
        pass
    low = src.lower()
    return not (low.startswith('rtsp://') or low.startswith('rtmp://')
                or low.startswith('http://') or low.startswith('https://'))


def probe_source(url: str, timeout: float = 5.0) -> dict:
    try:
        parsed = _parse_source(url)
        if isinstance(parsed, str) and parsed.lower().startswith('rtsp://'):
            cap = cv2.VideoCapture(parsed, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(parsed)

        if not cap.isOpened():
            cap.release()
            return {'ok': False, 'fps': None, 'error': 'Не удалось открыть источник'}

        t0 = time.time()
        ok = False
        fps = None
        while time.time() - t0 < timeout:
            ret, _ = cap.read()
            if ret:
                ok  = True
                fps = cap.get(cv2.CAP_PROP_FPS) or None
                break
            time.sleep(0.05)
        cap.release()
        return {'ok': ok, 'fps': fps, 'error': '' if ok else 'Нет кадров от источника'}
    except Exception as exc:
        return {'ok': False, 'fps': None, 'error': str(exc)}