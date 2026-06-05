"""
rtsp_manager.py — управление видеопотоком с автоматическим восстановлением.

Исправленные баги:
  - change_source() теперь гарантированно прерывает внутренний цикл чтения
    через флаг _change_requested, а не сравнение строк (старый код не работал)
  - _try_open() теперь реагирует на stop()/change_source() во время ожидания
    первого кадра — не висит 10 сек при остановке сервера
  - stop() дожидается завершения потока с разумным таймаутом и потом делает
    принудительный release() VideoCapture — поток больше не продолжает
    тянуть RTSP после Ctrl+C
  - Видеофайл зацикливается только если не было смены источника
"""

import logging
import threading
import time
from typing import Optional

import cv2

logger = logging.getLogger(__name__)

MAX_RECONNECT_DELAY  = 30
FIRST_FRAME_TIMEOUT  = 8    # сек — сколько ждём первый кадр при подключении


class RTSPManager:

    STATUS_CONNECTING   = 'connecting'
    STATUS_CONNECTED    = 'connected'
    STATUS_RECONNECTING = 'reconnecting'
    STATUS_STOPPED      = 'stopped'
    STATUS_NO_SOURCE    = 'no_source'   # источник не выбран — ничего не подключаем

    def __init__(self, source: str = ''):
        self._source_str = source or ''

        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock   = threading.Lock()      # защита _cap от race condition

        self._last_frame = None
        self._frame_seq  = 0          # увеличивается с каждым новым кадром
        self._frame_lock = threading.Lock()

        self._status         = self.STATUS_STOPPED
        self._error_msg      = ''
        self._reconnect_attempts = 0
        self._connected_at: Optional[float] = None
        self.stream_fps: float = 25.0

        # Флаг «живой»
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Событие — будит поток из любого sleep/wait немедленно
        self._wakeup = threading.Event()

        # Флаг смены источника — основной механизм прерывания цикла чтения
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
        logger.info('RTSPManager: запущен [%s]', self._source_str)

    def stop(self):
        """
        Останавливает поток и освобождает VideoCapture.
        Гарантирует, что после возврата никаких сетевых соединений нет.
        """
        self._running = False
        self._wakeup.set()          # прерываем любой sleep/wait в потоке

        # Принудительно освобождаем cap — это прерывает блокирующий cap.read()
        # в потоке (read вернёт False немедленно после release)
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
        """Последний кадр (BGR numpy) или None — не блокирует."""
        with self._frame_lock:
            return self._last_frame.copy() if self._last_frame is not None else None

    def read_seq(self):
        """
        Возвращает (seq, frame): порядковый номер кадра и сам кадр.
        Позволяет потребителю обрабатывать только новые кадры и не гонять
        YOLO по одному и тому же кадру многократно.
        """
        with self._frame_lock:
            if self._last_frame is None:
                return self._frame_seq, None
            return self._frame_seq, self._last_frame.copy()

    def change_source(self, new_source: str):
        """
        Горячая замена источника. Текущее соединение рвётся немедленно,
        поток переподключается к новому источнику.
        """
        new_source = '' if _is_empty(new_source) else str(new_source).strip()
        with self._source_lock:
            if self._source_str == new_source:
                return
            self._source_str       = new_source
            self._change_requested = True

        # Освобождаем текущий cap — прерывает блокирующий cap.read()
        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

        self._wakeup.set()   # будим поток если он спит между попытками
        logger.info('RTSPManager: источник изменён -> %s', new_source)

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

    # ------------------------------------------------------------------
    # Основной цикл (в отдельном потоке)
    # ------------------------------------------------------------------

    def _loop(self):
        reconnect_delay = 1.0

        while self._running:
            self._wakeup.clear()

            # Снимаем флаг смены источника и читаем текущий источник атомарно
            with self._source_lock:
                self._change_requested = False
                source_str = self._source_str

            # Источник не выбран — не подключаемся ни к чему, ждём выбора.
            # Это убирает «вечные попытки подключения к RTSP» при старте.
            if _is_empty(source_str):
                self._status = self.STATUS_NO_SOURCE
                self._error_msg = ''
                self._reconnect_attempts = 0
                reconnect_delay = 1.0
                self._connected_at = None
                with self._frame_lock:
                    self._last_frame = None      # убираем стоп-кадр прошлого источника
                self._wakeup.wait()              # спим до change_source()/stop()
                continue

            source = _parse_source(source_str)
            self._status = self.STATUS_CONNECTING
            logger.info('RTSPManager: подключение к [%s]', source_str)

            cap = self._try_open(source)

            if cap is None:
                # Подключиться не удалось
                self._reconnect_attempts += 1
                self._status    = self.STATUS_RECONNECTING
                self._error_msg = f'Нет сигнала: {source_str}'
                logger.warning('RTSPManager: попытка %d провалена, пауза %.0f с',
                               self._reconnect_attempts, reconnect_delay)
                self._wakeup.wait(timeout=reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)
                continue

            # Подключились
            with self._cap_lock:
                # Если stop() или change_source() уже вызвали release — cap
                # снаружи уже освобождён, выходим
                if not self._running or self._change_requested:
                    cap.release()
                    continue
                self._cap = cap

            raw_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            # Защита от некорректного значения FPS у файла/потока
            self.stream_fps         = raw_fps if 1.0 <= raw_fps <= 120.0 else 25.0
            self._status            = self.STATUS_CONNECTED
            self._connected_at      = time.time()
            self._error_msg         = ''
            reconnect_delay         = 1.0
            self._reconnect_attempts = 0
            is_file = _is_file(source_str)

            # Интервал между кадрами для пейсинга видеофайлов.
            # Камеры (USB/RTSP) сами выдают кадры в реальном времени, поэтому
            # для них пейсинг не нужен. Для файла OpenCV отдаёт кадры так быстро,
            # как успевает их декодировать — без пейсинга видео идёт «в ускоренном
            # режиме». Здесь привязываем чтение к собственному FPS файла.
            frame_interval = 1.0 / self.stream_fps if (is_file and self.stream_fps > 0) else 0.0
            next_frame_t   = time.time()
            slow_logged    = 0.0   # антиспам для предупреждений о падении FPS

            logger.info('RTSPManager: подключено FPS=%.1f [%s] (пейсинг=%s)',
                        self.stream_fps, source_str, 'вкл' if is_file else 'выкл')

            consecutive_failures = 0
            while self._running and not self._change_requested:
                ret, frame = cap.read()

                if not ret:
                    if is_file:
                        # Видеофайл закончился — зацикливаем,
                        # но только если источник не менялся
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

                # Пейсинг воспроизведения видеофайла под его собственный FPS.
                # Используем абсолютную метку времени следующего кадра, чтобы
                # ошибка не накапливалась. Прерываемся через _wakeup при
                # остановке/смене источника.
                if frame_interval > 0.0:
                    next_frame_t += frame_interval
                    delay = next_frame_t - time.time()
                    if delay > 0:
                        self._wakeup.wait(timeout=delay)
                    elif delay < -1.0:
                        # Декодер отстаёт более чем на секунду — система не
                        # успевает обрабатывать кадры в реальном времени.
                        now = time.time()
                        if now - slow_logged > 10.0:
                            logger.warning(
                                'RTSPManager: обработка отстаёт от FPS файла '
                                '(%.1f FPS), кадры могут пропускаться', self.stream_fps)
                            slow_logged = now
                        next_frame_t = now   # сбрасываем дрейф

            # Выходим из цикла чтения — освобождаем cap
            with self._cap_lock:
                if self._cap is cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    self._cap = None

        self._status = self.STATUS_STOPPED

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _try_open(self, source) -> Optional[cv2.VideoCapture]:
        """
        Открывает VideoCapture и ждёт первый кадр.
        Прерывается мгновенно если stop() или change_source() вызваны
        во время ожидания.
        """
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
                # Прерываемся если сервер останавливается или источник сменился
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
# Утилиты (модульного уровня — не засоряют класс)
# ------------------------------------------------------------------

def _is_empty(src) -> bool:
    """True если источник не задан (пусто/none) — подключаться не нужно."""
    return src is None or str(src).strip().lower() in ('', 'none', 'null')


def _parse_source(src: str):
    """'0' -> int(0), иначе строка."""
    try:
        return int(src)
    except (ValueError, TypeError):
        return src


def _is_file(src: str) -> bool:
    """True если источник — локальный файл (не камера и не RTSP)."""
    try:
        int(src)
        return False   # целое число = устройство
    except (ValueError, TypeError):
        pass
    low = src.lower()
    return not (low.startswith('rtsp://') or low.startswith('rtmp://')
                or low.startswith('http://') or low.startswith('https://'))


def probe_source(url: str, timeout: float = 5.0) -> dict:
    """
    Быстрая проверка доступности источника.
    Возвращает {'ok': bool, 'fps': float|None, 'error': str}.
    """
    try:
        parsed = _parse_source(url)
        if isinstance(parsed, str) and parsed.lower().startswith('rtsp://'):
            cap = cv2.VideoCapture(parsed, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(parsed)

        if not cap.isOpened():
            cap.release()
            return {'ok': False, 'fps': None, 'error': 'Не удалось открыть источник'}

        t0  = time.time()
        ok  = False
        fps = None
        while time.time() - t0 < timeout:
            ret, _ = cap.read()
            if ret:
                ok  = True
                fps = cap.get(cv2.CAP_PROP_FPS) or None
                break
            time.sleep(0.05)
        cap.release()
        return {'ok': ok, 'fps': fps,
                'error': '' if ok else 'Нет кадров от источника'}
    except Exception as exc:
        return {'ok': False, 'fps': None, 'error': str(exc)}
