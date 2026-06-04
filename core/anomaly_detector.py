"""
anomaly_detector.py — конечный автомат для обнаружения аномалий на АЗС.

Обнаруживаемые аномалии:
  1. Человек без автомобиля  — person в кадре, car отсутствует > 60 сек
  2. Долгое нахождение       — person непрерывно в кадре > 600 сек
  3. Автомобиль без человека — car стоит в кадре, person отсутствует > 300 сек

Каждый трек (track_id) хранит собственное состояние.
Для глобальных условий (например, «car в кадре») используется агрегированный
взгляд на все активные треки, а не отдельный трек.

Таймауты настраиваются через update_settings() без перезапуска.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Структура состояния одного трека
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    track_id:       int
    cls:            str
    first_seen:     float = field(default_factory=time.time)
    last_seen:      float = field(default_factory=time.time)
    # Флаг: в момент последнего кадра рядом был автомобиль
    has_car_in_frame: bool = False
    # Время, когда человек вошёл в кадр без автомобиля
    alone_since:    Optional[float] = None
    # Флаги, чтобы не дублировать тревоги по одному треку
    alert_alone_sent:   bool = False
    alert_long_sent:    bool = False


@dataclass
class CarTrackState:
    """Состояние трека автомобиля для аномалии 'автомобиль без человека'."""
    track_id:   int
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)
    # Время, когда автомобиль остался без человека рядом
    alone_since: Optional[float] = None
    alert_sent:  bool = False


# ---------------------------------------------------------------------------
# Конечный автомат аномалий
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Принимает на вход список детекций каждого кадра и генерирует события-аномалии.

    Использование:
        detector = AnomalyDetector()
        # в цикле обработки кадров:
        alerts = detector.process(detections)
        for alert in alerts:
            save_event(alert['type'], alert['obj'], frame)
    """

    def __init__(
        self,
        # Таймаут для аномалии «человек без автомобиля», секунд
        person_alone_timeout: float = 60.0,
        # Cooldown между повторными тревогами «человек без авто»
        person_alone_cooldown: float = 60.0,
        # Таймаут для аномалии «долгое нахождение человека», секунд
        person_long_stay_timeout: float = 600.0,
        # Cooldown между повторными тревогами «долгое нахождение»
        person_long_cooldown: float = 300.0,
        # Таймаут для аномалии «автомобиль без человека», секунд
        car_alone_timeout: float = 300.0,
        # Cooldown между повторными тревогами «авто без человека»
        car_alone_cooldown: float = 120.0,
        # Сколько секунд трек «помнится» после исчезновения из кадра
        track_memory_seconds: float = 10.0,
    ):
        self.person_alone_timeout    = person_alone_timeout
        self.person_alone_cooldown   = person_alone_cooldown
        self.person_long_stay_timeout = person_long_stay_timeout
        self.person_long_cooldown    = person_long_cooldown
        self.car_alone_timeout       = car_alone_timeout
        self.car_alone_cooldown      = car_alone_cooldown
        self.track_memory_seconds    = track_memory_seconds

        # Активные треки людей
        self._persons: dict[int, TrackState] = {}
        # Активные треки автомобилей (car/bus/truck)
        self._cars: dict[int, CarTrackState] = {}

        # Время последних глобальных тревог (антидребезг по типу)
        self._last_alert_time: dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Публичный интерфейс
    # -----------------------------------------------------------------------

    def update_settings(self, **kwargs):
        """Обновляет параметры детектора без перезапуска."""
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, float(val))
                logger.debug('AnomalyDetector: %s = %s', key, val)

    def process(self, detections: list, now: Optional[float] = None) -> list[dict]:
        """
        Обрабатывает список детекций текущего кадра.

        Параметры:
            detections — список словарей от Detector.process_frame()
            now        — текущее время (если None — берётся time.time())

        Возвращает список тревог:
            [{'type': str, 'obj': dict, 'description': str}, ...]
        """
        if now is None:
            now = time.time()

        alerts = []

        # Разбиваем детекции по категориям
        person_detections = [d for d in detections if d.get('class') == 'person']
        car_detections    = [d for d in detections
                             if d.get('class') in ('car', 'bus', 'truck')]

        # Флаги присутствия в кадре (используются в нескольких правилах)
        persons_present = len(person_detections) > 0
        cars_present    = len(car_detections) > 0

        # --- Обновляем состояния треков людей ---
        active_person_ids = set()
        for obj in person_detections:
            tid = obj.get('track_id')
            if tid is None:
                continue
            active_person_ids.add(tid)
            self._update_person(tid, obj, cars_present, now)

        # --- Обновляем состояния треков автомобилей ---
        active_car_ids = set()
        for obj in car_detections:
            tid = obj.get('track_id')
            if tid is None:
                continue
            active_car_ids.add(tid)
            self._update_car(tid, obj, persons_present, now)

        # --- Проверяем аномалии для людей ---
        for tid, state in list(self._persons.items()):
            # Трек исчез из кадра — обновляем last_seen
            if tid not in active_person_ids:
                if now - state.last_seen > self.track_memory_seconds:
                    # Трек «забыт» — сбрасываем состояние
                    self._forget_person(tid)
                    continue

            alert = self._check_person_anomalies(state, now)
            if alert:
                alerts.append(alert)

        # --- Проверяем аномалии для автомобилей ---
        for tid, state in list(self._cars.items()):
            if tid not in active_car_ids:
                if now - state.last_seen > self.track_memory_seconds:
                    self._forget_car(tid)
                    continue

            alert = self._check_car_anomalies(state, now)
            if alert:
                alerts.append(alert)

        return alerts

    def reset(self):
        """Полный сброс всех состояний (при остановке детекции)."""
        self._persons.clear()
        self._cars.clear()
        self._last_alert_time.clear()
        logger.info('AnomalyDetector: состояние сброшено')

    def get_active_states(self) -> dict:
        """Возвращает текущие состояния для отладки / отображения в UI."""
        now = time.time()
        return {
            'persons': [
                {
                    'track_id':    s.track_id,
                    'cls':         s.cls,
                    'in_frame_sec': round(now - s.first_seen, 1),
                    'alone_sec':   round(now - s.alone_since, 1) if s.alone_since else None,
                    'alert_alone': s.alert_alone_sent,
                    'alert_long':  s.alert_long_sent,
                }
                for s in self._persons.values()
            ],
            'cars': [
                {
                    'track_id':   s.track_id,
                    'alone_sec':  round(now - s.alone_since, 1) if s.alone_since else None,
                    'alert_sent': s.alert_sent,
                }
                for s in self._cars.values()
            ],
        }

    # -----------------------------------------------------------------------
    # Внутренние методы — обновление состояния
    # -----------------------------------------------------------------------

    def _update_person(self, tid: int, obj: dict, cars_present: bool, now: float):
        """Обновляет или создаёт состояние трека человека."""
        if tid not in self._persons:
            self._persons[tid] = TrackState(
                track_id=tid,
                cls='person',
                first_seen=now,
                last_seen=now,
            )
            logger.debug('Новый трек человека: ID=%d', tid)
        state = self._persons[tid]
        state.last_seen = now
        state.cls = obj.get('class', 'person')
        state.has_car_in_frame = cars_present

        # Логика таймера «один в кадре»
        if not cars_present:
            if state.alone_since is None:
                state.alone_since = now
                logger.debug('Человек ID=%d остался без авто', tid)
        else:
            # Автомобиль появился — сбрасываем таймер одиночества
            if state.alone_since is not None:
                logger.debug('Человек ID=%d: авто появилось, таймер сброшен', tid)
            state.alone_since = None
            state.alert_alone_sent = False

    def _update_car(self, tid: int, obj: dict, persons_present: bool, now: float):
        """Обновляет или создаёт состояние трека автомобиля."""
        if tid not in self._cars:
            self._cars[tid] = CarTrackState(
                track_id=tid,
                first_seen=now,
                last_seen=now,
            )
            logger.debug('Новый трек авто: ID=%d', tid)
        state = self._cars[tid]
        state.last_seen = now

        if not persons_present:
            if state.alone_since is None:
                state.alone_since = now
                logger.debug('Авто ID=%d: люди ушли из кадра', tid)
        else:
            # Человек появился рядом — сбрасываем
            if state.alone_since is not None:
                logger.debug('Авто ID=%d: человек вернулся, таймер сброшен', tid)
            state.alone_since = None
            state.alert_sent = False

    def _forget_person(self, tid: int):
        state = self._persons.pop(tid, None)
        if state:
            logger.debug('Трек человека ID=%d забыт (исчез из кадра)', tid)

    def _forget_car(self, tid: int):
        state = self._cars.pop(tid, None)
        if state:
            logger.debug('Трек авто ID=%d забыт', tid)

    # -----------------------------------------------------------------------
    # Внутренние методы — проверка аномалий
    # -----------------------------------------------------------------------

    def _check_person_anomalies(self, state: TrackState, now: float) -> Optional[dict]:
        """
        Проверяет аномалии для конкретного трека человека.
        Возвращает словарь тревоги или None.
        """
        # Аномалия 1: человек без автомобиля
        if (state.alone_since is not None
                and not state.alert_alone_sent
                and now - state.alone_since >= self.person_alone_timeout):

            if self._cooldown_ok('person_alone', self.person_alone_cooldown, now):
                state.alert_alone_sent = True
                duration = round(now - state.alone_since)
                logger.info(
                    'АНОМАЛИЯ: человек без авто ID=%d, %d сек', state.track_id, duration
                )
                return {
                    'type': 'Человек без автомобиля',
                    'obj':  self._make_obj(state),
                    'description': (
                        f'Человек (ID {state.track_id}) находится без автомобиля '
                        f'{duration} секунд.'
                    ),
                }

        # Аномалия 2: долгое нахождение человека в кадре
        elapsed = now - state.first_seen
        if (not state.alert_long_sent
                and elapsed >= self.person_long_stay_timeout):

            if self._cooldown_ok('person_long', self.person_long_cooldown, now):
                state.alert_long_sent = True
                logger.info(
                    'АНОМАЛИЯ: долгое нахождение человека ID=%d, %.0f сек',
                    state.track_id, elapsed
                )
                return {
                    'type': 'Долгое нахождение человека',
                    'obj':  self._make_obj(state),
                    'description': (
                        f'Человек (ID {state.track_id}) непрерывно в кадре '
                        f'{round(elapsed)} секунд.'
                    ),
                }

        return None

    def _check_car_anomalies(self, state: CarTrackState, now: float) -> Optional[dict]:
        """Проверяет аномалии для конкретного трека автомобиля."""
        if (state.alone_since is not None
                and not state.alert_sent
                and now - state.alone_since >= self.car_alone_timeout):

            if self._cooldown_ok('car_alone', self.car_alone_cooldown, now):
                state.alert_sent = True
                duration = round(now - state.alone_since)
                logger.info(
                    'АНОМАЛИЯ: авто без человека ID=%d, %d сек', state.track_id, duration
                )
                return {
                    'type': 'Автомобиль без обслуживания',
                    'obj':  {'track_id': state.track_id, 'class': 'car',
                             'confidence': 0.9, 'bbox': []},
                    'description': (
                        f'Автомобиль (ID {state.track_id}) стоит без обслуживания '
                        f'{duration} секунд.'
                    ),
                }

        return None

    def _cooldown_ok(self, key: str, cooldown: float, now: float) -> bool:
        """
        Проверяет антидребезг: возвращает True и обновляет таймер,
        если с последней тревоги данного типа прошло достаточно времени.
        """
        last = self._last_alert_time.get(key, 0.0)
        if now - last >= cooldown:
            self._last_alert_time[key] = now
            return True
        return False

    @staticmethod
    def _make_obj(state: TrackState) -> dict:
        """Формирует словарь объекта для передачи в save_event()."""
        return {
            'track_id':   state.track_id,
            'class':      state.cls,
            'confidence': 0.9,
            'bbox':       [],
        }
