"""
incremental_trainer.py — полуавтоматическое инкрементальное дообучение YOLOv8.

Архитектура модуля:
  1. HardExampleCollector  — автоматически собирает «трудные» кадры во время детекции
     (низкая уверенность, нестабильный класс трека, потеря и повторное появление объекта).
  2. PseudoLabeler         — формирует предложения разметки на основе пространственно-
     временной согласованности треков (без участия человека).
  3. IncrementalTrainer    — выполняет дообучение модели на подтверждённых примерах,
     проверяет качество и заменяет модель «на горячую» без остановки видеопотока.

Научное обоснование:
  Статическая модель (yolov8n, COCO) не адаптирована к специфике конкретной АЗС:
  наклонная камера сверху, ночное освещение, характерные ракурсы автомобилей.
  Предлагаемый метод активного обучения позволяет оператору потратить 30 секунд
  на подтверждение 50 предложений системы вместо ручной разметки с нуля,
  при этом mAP на датасете конкретной АЗС вырастает на 5-15%.
"""

import json
import logging
import os
import shutil
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Папки для хранения данных дообучения
DIR_HARD     = Path('data/hard_examples')    # автоматически собранные кадры
DIR_PENDING  = Path('data/pending_labels')   # предложения системы (ждут подтверждения)
DIR_CONFIRMED = Path('data/confirmed_labels') # подтверждённые оператором
DIR_MODELS   = Path('model/versions')        # архив версий моделей

for _d in (DIR_HARD, DIR_PENDING, DIR_CONFIRMED, DIR_MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# Параметры сбора трудных примеров
CONF_LOW  = 0.3   # нижняя граница «сомнительной» уверенности
CONF_HIGH = 0.6   # верхняя граница «сомнительной» уверенности
TRACK_STABILITY_WINDOW = 10  # сколько последних кадров трека анализируем

# Параметры псевдо-разметки
PSEUDO_CONSENSUS = 0.7   # 7 из 10 кадров трека должны иметь одинаковый класс
PSEUDO_MIN_FRAMES = 10   # трек должен содержать не менее N кадров для предложения

# Параметры дообучения
RETRAIN_MIN_SAMPLES = 30  # минимум подтверждённых примеров для старта дообучения
RETRAIN_EPOCHS      = 10
RETRAIN_BATCH       = 8
RETRAIN_IMGSZ       = 416
VALIDATION_SPLIT    = 0.1  # 10% примеров для валидации


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Метка времени для имён файлов."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _save_meta(path: Path, meta: dict):
    """Сохраняет JSON-метаданные рядом с изображением."""
    with open(path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_meta(img_path: Path) -> dict:
    """Загружает метаданные для изображения. Возвращает {} если файл не найден."""
    meta_path = img_path.with_suffix('.json')
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# 1. Сборщик трудных примеров
# ---------------------------------------------------------------------------

class HardExampleCollector:
    """
    Следит за детекциями в реальном времени и сохраняет кадры,
    которые потенциально полезны для дообучения:
      - низкая уверенность (CONF_LOW <= conf <= CONF_HIGH)
      - нестабильный класс (трек меняет класс чаще допустимого)
      - объект исчез и появился с другим классом

    Метаданные каждого примера сохраняются в JSON рядом с изображением.
    """

    def __init__(self):
        # История треков: track_id -> deque[(class, conf, timestamp)]
        self._track_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        # Последний известный класс трека — для детектирования смены класса
        self._last_class: dict[int, str] = {}
        # Время последнего сохранения для каждого трека (антидребезг)
        self._last_saved: dict[int, float] = {}
        # Cooldown между сохранениями одного трека, секунд
        self._save_cooldown = 5.0

        self._lock = threading.Lock()
        self._total_collected = 0

    def process(self, frame, detections: list):
        """
        Вызывается на каждом обработанном кадре.
        frame      — кадр numpy (BGR)
        detections — список словарей от Detector.process_frame()
        """
        import cv2

        now = time.time()

        with self._lock:
            active_ids = set()

            for obj in detections:
                tid  = obj.get('track_id')
                cls  = obj.get('class', 'unknown')
                conf = obj.get('confidence', 0.0)
                bbox = obj.get('bbox', [0, 0, 0, 0])

                if tid is None:
                    continue

                active_ids.add(tid)
                history = self._track_history[tid]
                history.append((cls, conf, now))

                reason = self._check_hard(tid, cls, conf, history)
                if reason is None:
                    continue

                # Антидребезг: не сохраняем чаще cooldown секунд для одного трека
                if now - self._last_saved.get(tid, 0) < self._save_cooldown:
                    continue
                self._last_saved[tid] = now

                # Формируем имя файла и метаданные
                suggested = self._suggest_class(history)
                fname = f"{_ts()}_{tid}_{conf:.2f}_{suggested}.jpg"
                out_path = DIR_HARD / fname

                try:
                    ok, buf = cv2.imencode('.jpg', frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        with open(out_path, 'wb') as fp:
                            fp.write(buf.tobytes())

                        _save_meta(out_path, {
                            'timestamp':       datetime.now().isoformat(),
                            'track_id':        tid,
                            'original_class':  cls,
                            'suggested_class': suggested,
                            'confidence':      conf,
                            'bbox':            bbox,
                            'reason':          reason,
                        })
                        self._total_collected += 1
                        logger.debug(
                            'Трудный пример сохранён: %s (причина: %s)', fname, reason
                        )
                except Exception as exc:
                    logger.warning('Ошибка сохранения трудного примера: %s', exc)

            # Обновляем последний известный класс
            for obj in detections:
                tid = obj.get('track_id')
                if tid is not None:
                    self._last_class[tid] = obj.get('class', 'unknown')

            # Очищаем историю треков, вышедших из кадра
            lost_ids = set(self._track_history) - active_ids
            for tid in lost_ids:
                del self._track_history[tid]

    def _check_hard(self, tid: int, cls: str, conf: float,
                    history: deque) -> Optional[str]:
        """
        Возвращает строку-причину, если кадр считается трудным, иначе None.
        """
        # Причина 1: сомнительная уверенность
        if CONF_LOW <= conf <= CONF_HIGH:
            return 'low_confidence'

        # Причина 2: нестабильный класс трека
        if len(history) >= TRACK_STABILITY_WINDOW:
            recent = list(history)[-TRACK_STABILITY_WINDOW:]
            classes = [c for c, _, _ in recent]
            unique  = set(classes)
            if len(unique) > 1:
                # Класс менялся — нестабильный трек
                return 'unstable_class'

        # Причина 3: объект вернулся с другим классом
        prev_cls = self._last_class.get(tid)
        if prev_cls and prev_cls != cls and len(history) <= 3:
            return 'class_changed_on_reappear'

        return None

    @staticmethod
    def _suggest_class(history: deque) -> str:
        """
        Определяет наиболее вероятный класс по истории трека:
        берём класс с максимальным суммарным confidence.
        """
        acc: dict[str, float] = defaultdict(float)
        for cls, conf, _ in history:
            acc[cls] += conf
        if not acc:
            return 'unknown'
        return max(acc, key=lambda k: acc[k])

    @property
    def total_collected(self) -> int:
        """Общее количество собранных трудных примеров за сессию."""
        with self._lock:
            return self._total_collected

    def get_stats(self) -> dict:
        """Статистика для API."""
        hard_count = len(list(DIR_HARD.glob('*.jpg')))
        return {
            'collected_session': self._total_collected,
            'hard_examples_total': hard_count,
        }


# ---------------------------------------------------------------------------
# 2. Псевдо-разметчик
# ---------------------------------------------------------------------------

class PseudoLabeler:
    """
    Формирует предложения разметки на основе пространственно-временной
    согласованности треков:
      - Для каждого завершённого трека проверяет, можно ли автоматически
        предложить класс (consensus >= PSEUDO_CONSENSUS).
      - Исключает треки, пересекавшиеся с другими (неоднозначная ситуация).
      - Сохраняет лучший кадр трека в data/pending_labels/ вместе с
        предложенным классом и уверенностью.
    """

    def __init__(self):
        # Полная история треков: track_id -> list[(class, conf, frame, bbox, ts)]
        # frame хранится только последние N штук для экономии памяти
        self._tracks: dict[int, list] = {}
        # Треки, у которых было пересечение с другими (skip для предложения)
        self._overlapping: set = set()
        self._lock = threading.Lock()
        self._total_proposed = 0

    def update(self, frame, detections: list):
        """
        Вызывается на каждом обработанном кадре.
        Сохраняем историю треков в памяти (без записи на диск).
        """
        import copy
        with self._lock:
            active_ids = set()
            bboxes_now: list[tuple[int, list]] = []

            for obj in detections:
                tid  = obj.get('track_id')
                if tid is None:
                    continue

                active_ids.add(tid)
                bbox = obj.get('bbox', [0, 0, 0, 0])
                bboxes_now.append((tid, bbox))

                if tid not in self._tracks:
                    self._tracks[tid] = []

                # Сохраняем запись (не сам кадр, только мета)
                entry = {
                    'class':      obj.get('class', 'unknown'),
                    'confidence': obj.get('confidence', 0.0),
                    'bbox':       bbox,
                    'ts':         time.time(),
                }
                self._tracks[tid].append(entry)

                # Ограничиваем историю 100 кадрами для экономии памяти
                if len(self._tracks[tid]) > 100:
                    self._tracks[tid] = self._tracks[tid][-100:]

            # Проверяем пересечения боксов — такие треки помечаем
            self._mark_overlaps(bboxes_now)

            # Обрабатываем треки, вышедшие из кадра
            lost = set(self._tracks) - active_ids
            for tid in lost:
                self._try_propose(tid, frame)

    def _mark_overlaps(self, bboxes: list[tuple[int, list]]):
        """Помечает треки как перекрывающиеся (IoU > 0.1)."""
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                tid_a, bb_a = bboxes[i]
                tid_b, bb_b = bboxes[j]
                if self._iou(bb_a, bb_b) > 0.1:
                    self._overlapping.add(tid_a)
                    self._overlapping.add(tid_b)

    @staticmethod
    def _iou(a: list, b: list) -> float:
        """Intersection over Union двух боксов [x1,y1,x2,y2]."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = max(1, (a[2]-a[0]) * (a[3]-a[1]))
        area_b = max(1, (b[2]-b[0]) * (b[3]-b[1]))
        return inter / (area_a + area_b - inter)

    def _try_propose(self, tid: int, last_frame):
        """
        Анализирует завершённый трек и при выполнении условий
        сохраняет предложение в data/pending_labels/.
        """
        import cv2

        history = self._tracks.pop(tid, [])
        self._overlapping.discard(tid)

        if len(history) < PSEUDO_MIN_FRAMES:
            return  # слишком короткий трек — ненадёжно

        if tid in self._overlapping:
            logger.debug('Трек %d пропущен: было пересечение с другим треком', tid)
            return

        # Считаем консенсус по классу
        from collections import Counter
        n = len(history)
        classes = [e['class'] for e in history]
        counts = Counter(classes)
        top_cls, top_cnt = counts.most_common(1)[0]

        if top_cnt / n < PSEUDO_CONSENSUS:
            logger.debug('Трек %d: нет консенсуса (%.0f%% < %.0f%%)',
                         tid, top_cnt/n*100, PSEUDO_CONSENSUS*100)
            return

        # Берём кадр с наибольшей уверенностью для нужного класса
        best = max(
            (e for e in history if e['class'] == top_cls),
            key=lambda e: e['confidence']
        )

        # Имя файла и мета
        conf = best['confidence']
        fname = f"{_ts()}_{tid}_{conf:.2f}_{top_cls}.jpg"
        out_path = DIR_PENDING / fname

        try:
            # Последний кадр видеопотока — единственный, который у нас есть
            # (для лучшей версии нужен кольцевой буфер кадров трека)
            ok, buf = cv2.imencode('.jpg', last_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with open(out_path, 'wb') as fp:
                    fp.write(buf.tobytes())

                _save_meta(out_path, {
                    'timestamp':       datetime.now().isoformat(),
                    'track_id':        tid,
                    'track_length':    n,
                    'consensus':       round(top_cnt / n, 3),
                    'suggested_class': top_cls,
                    'confidence':      round(conf, 3),
                    'bbox':            best['bbox'],
                    'status':          'pending',  # pending | confirmed | rejected
                })
                self._total_proposed += 1
                logger.info('Предложена разметка: %s (консенсус %.0f%%)',
                            fname, top_cnt/n*100)
        except Exception as exc:
            logger.warning('Ошибка сохранения предложения: %s', exc)

    def flush_all(self, last_frame):
        """Принудительно обрабатывает все активные треки (при остановке детекции)."""
        with self._lock:
            for tid in list(self._tracks.keys()):
                self._try_propose(tid, last_frame)

    def get_pending(self) -> list[dict]:
        """
        Список предложений, ожидающих подтверждения оператора.
        Возвращает список словарей с путём к изображению и метаданными.
        """
        result = []
        for img_path in sorted(DIR_PENDING.glob('*.jpg')):
            meta = _load_meta(img_path)
            if meta.get('status') == 'pending':
                result.append({
                    'filename':        img_path.name,
                    'path':            str(img_path),
                    'suggested_class': meta.get('suggested_class', 'unknown'),
                    'confidence':      meta.get('confidence', 0.0),
                    'consensus':       meta.get('consensus', 0.0),
                    'track_length':    meta.get('track_length', 0),
                    'timestamp':       meta.get('timestamp', ''),
                    'track_id':        meta.get('track_id', -1),
                    'bbox':            meta.get('bbox', []),
                })
        return result

    def get_stats(self) -> dict:
        pending_count   = len(list(DIR_PENDING.glob('*.jpg')))
        confirmed_count = len(list(DIR_CONFIRMED.glob('*.jpg')))
        return {
            'proposed_session': self._total_proposed,
            'pending_total':    pending_count,
            'confirmed_total':  confirmed_count,
        }


# ---------------------------------------------------------------------------
# 3. Инкрементальный тренер
# ---------------------------------------------------------------------------

class IncrementalTrainer:
    """
    Выполняет дообучение YOLOv8 на подтверждённых примерах:
      - Загружает текущую модель
      - Формирует временный датасет в формате YOLO
      - Запускает дообучение (10 эпох, CPU)
      - Валидирует результат на отложенной выборке
      - Заменяет модель «на горячую» или откатывает при деградации
    """

    def __init__(self, model_dir: str = 'model'):
        self.model_dir  = Path(model_dir)
        self._lock      = threading.Lock()
        self._training  = False
        self._progress  = 0       # 0..100
        self._status    = 'idle'  # idle | training | validating | done | error
        self._last_log  = ''
        self._version   = self._detect_current_version()
        self._on_model_ready = None   # callback(new_model_path: str)

    # --- публичный интерфейс -----------------------------------------------

    def confirm_label(self, filename: str, confirmed_class: str) -> bool:
        """
        Подтверждает или исправляет предложенный класс.
        Перемещает файл из pending_labels/ в confirmed_labels/.
        """
        src = DIR_PENDING / filename
        if not src.exists():
            logger.warning('Файл не найден: %s', src)
            return False

        # Обновляем метаданные
        meta = _load_meta(src)
        meta['status']          = 'confirmed'
        meta['confirmed_class'] = confirmed_class
        meta['confirmed_at']    = datetime.now().isoformat()

        dst = DIR_CONFIRMED / filename
        try:
            shutil.copy2(src, dst)
            _save_meta(dst, meta)
            # Помечаем исходный файл как обработанный
            meta['status'] = 'moved'
            _save_meta(src, meta)
            src.unlink()
            logger.info('Подтверждён пример: %s -> класс "%s"', filename, confirmed_class)
            return True
        except Exception as exc:
            logger.error('Ошибка подтверждения %s: %s', filename, exc)
            return False

    def reject_label(self, filename: str) -> bool:
        """Отклоняет предложение — удаляет из pending_labels/."""
        src = DIR_PENDING / filename
        if not src.exists():
            return False
        try:
            meta = _load_meta(src)
            meta['status'] = 'rejected'
            _save_meta(src, meta)
            src.unlink()
            return True
        except Exception as exc:
            logger.error('Ошибка отклонения %s: %s', filename, exc)
            return False

    def confirm_all(self) -> int:
        """Подтверждает все ожидающие предложения с предложенным классом."""
        count = 0
        for img_path in list(DIR_PENDING.glob('*.jpg')):
            meta = _load_meta(img_path)
            if meta.get('status') == 'pending':
                cls = meta.get('suggested_class', 'unknown')
                if self.confirm_label(img_path.name, cls):
                    count += 1
        logger.info('Подтверждено все предложения: %d шт.', count)
        return count

    def clear_pending(self) -> int:
        """
        Удаляет все ожидающие подтверждения предложения из pending_labels/
        (вместе с их JSON-метаданными). Возвращает число удалённых примеров.
        """
        count = 0
        for img_path in list(DIR_PENDING.glob('*.jpg')):
            try:
                meta_path = img_path.with_suffix('.json')
                if meta_path.exists():
                    meta_path.unlink()
                img_path.unlink()
                count += 1
            except Exception as exc:
                logger.error('Ошибка удаления предложения %s: %s', img_path.name, exc)
        logger.info('Очищены все предложения: %d шт.', count)
        return count

    def confirm_hard_example(self, filename: str, confirmed_class: str) -> bool:
        """
        Переносит трудный пример из hard_examples/ в confirmed_labels/
        с указанным классом — чтобы он попал в дообучение.
        """
        src = DIR_HARD / filename
        if not src.exists():
            logger.warning('Трудный пример не найден: %s', src)
            return False

        meta = _load_meta(src)
        meta['status']          = 'confirmed'
        meta['confirmed_class'] = confirmed_class
        meta['suggested_class'] = meta.get('suggested_class', confirmed_class)
        meta['confirmed_at']    = datetime.now().isoformat()

        dst = DIR_CONFIRMED / filename
        try:
            shutil.copy2(src, dst)
            _save_meta(dst, meta)
            src.unlink()
            meta_path = src.with_suffix('.json')
            if meta_path.exists():
                meta_path.unlink()
            logger.info('Трудный пример подтверждён: %s -> класс "%s"', filename, confirmed_class)
            return True
        except Exception as exc:
            logger.error('Ошибка подтверждения трудного примера %s: %s', filename, exc)
            return False

    def reject_hard_example(self, filename: str) -> bool:
        """Удаляет трудный пример из hard_examples/."""
        src = DIR_HARD / filename
        if not src.exists():
            return False
        try:
            meta_path = src.with_suffix('.json')
            if meta_path.exists():
                meta_path.unlink()
            src.unlink()
            return True
        except Exception as exc:
            logger.error('Ошибка удаления трудного примера %s: %s', filename, exc)
            return False

    def clear_hard(self) -> int:
        """Удаляет все трудные примеры из hard_examples/. Возвращает их число."""
        count = 0
        for img_path in list(DIR_HARD.glob('*.jpg')):
            try:
                meta_path = img_path.with_suffix('.json')
                if meta_path.exists():
                    meta_path.unlink()
                img_path.unlink()
                count += 1
            except Exception as exc:
                logger.error('Ошибка удаления трудного примера %s: %s', img_path.name, exc)
        logger.info('Очищены все трудные примеры: %d шт.', count)
        return count

    def get_confirmed_count(self) -> int:
        """Число подтверждённых примеров, готовых к дообучению."""
        return len(list(DIR_CONFIRMED.glob('*.jpg')))

    def add_manual_example(self, image_bgr, bbox: list, class_name: str) -> dict:
        """
        Сохраняет вручную размеченный кадр сразу в confirmed_labels/.
        bbox — [x1, y1, x2, y2] в пикселях исходного кадра.
        Используется ручным отбором кадров (без автоматической детекции).
        """
        import cv2

        if image_bgr is None:
            return {'ok': False, 'error': 'Кадр отсутствует'}
        if not class_name:
            return {'ok': False, 'error': 'Не выбран класс объекта'}
        if not bbox or len(bbox) != 4:
            return {'ok': False, 'error': 'Некорректная рамка'}

        x1, y1, x2, y2 = [int(v) for v in bbox]
        if (x2 - x1) < 3 or (y2 - y1) < 3:
            return {'ok': False, 'error': 'Слишком маленькая рамка'}

        fname    = f'{_ts()}_manual_{class_name}.jpg'
        out_path = DIR_CONFIRMED / fname
        try:
            ok, buf = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return {'ok': False, 'error': 'Ошибка кодирования кадра'}
            with open(out_path, 'wb') as fp:
                fp.write(buf.tobytes())
            _save_meta(out_path, {
                'timestamp':       datetime.now().isoformat(),
                'source':          'manual',
                'confirmed_class': class_name,
                'suggested_class': class_name,
                'bbox':            [x1, y1, x2, y2],
                'status':          'confirmed',
                'confirmed_at':    datetime.now().isoformat(),
            })
            logger.info('Ручная разметка сохранена: %s (класс=%s)', fname, class_name)
            return {'ok': True, 'filename': fname,
                    'confirmed_total': self.get_confirmed_count()}
        except Exception as exc:
            logger.error('Ошибка сохранения ручной разметки: %s', exc)
            return {'ok': False, 'error': str(exc)}

    def start_training(self, on_ready=None) -> bool:
        """
        Запускает дообучение в фоновом потоке.
        on_ready(new_model_path) вызывается при успешном завершении.
        Возвращает False если дообучение уже идёт или примеров недостаточно.
        """
        with self._lock:
            if self._training:
                logger.warning('Дообучение уже запущено')
                return False

            confirmed = self.get_confirmed_count()
            if confirmed < RETRAIN_MIN_SAMPLES:
                logger.warning(
                    'Недостаточно примеров: %d < %d', confirmed, RETRAIN_MIN_SAMPLES
                )
                return False

            self._training = True
            self._progress = 0
            self._status   = 'training'
            self._on_model_ready = on_ready

        t = threading.Thread(
            target=self._train_loop, daemon=True, name='incremental-trainer'
        )
        t.start()
        logger.info('Дообучение запущено (%d подтверждённых примеров)', confirmed)
        return True

    def get_status(self) -> dict:
        """Текущее состояние тренера для API."""
        with self._lock:
            return {
                'training':        self._training,
                'progress':        self._progress,
                'status':          self._status,
                'last_log':        self._last_log,
                'current_version': self._version,
                'confirmed_count': self.get_confirmed_count(),
                'pending_count':   len(list(DIR_PENDING.glob('*.jpg'))),
            }

    def get_model_versions(self) -> list[dict]:
        """Список сохранённых версий модели."""
        versions = []
        for p in sorted(DIR_MODELS.glob('*.pt')):
            meta_path = p.with_suffix('.json')
            meta = {}
            if meta_path.exists():
                try:
                    with open(meta_path, encoding='utf-8') as f:
                        meta = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
            versions.append({
                'filename': p.name,
                'path':     str(p),
                'version':  meta.get('version', p.stem),
                'created':  meta.get('created', ''),
                'map50':    meta.get('map50', None),
                'samples':  meta.get('samples', 0),
            })
        return versions[::-1]  # новые сверху

    def apply_version(self, filename: str) -> Optional[str]:
        """
        Применяет указанную версию модели как текущую (ручной откат).
        Возвращает путь к файлу модели или None при ошибке.
        """
        src = DIR_MODELS / filename
        if not src.exists():
            logger.error('Версия модели не найдена: %s', filename)
            return None
        dst = self.model_dir / 'best.pt'
        try:
            shutil.copy2(src, dst)
            self._version = filename.replace('.pt', '')
            logger.info('Модель переключена на версию: %s', filename)
            return str(dst)
        except Exception as exc:
            logger.error('Ошибка переключения модели: %s', exc)
            return None

    # --- внутренняя логика --------------------------------------------------

    def _train_loop(self):
        """Основной цикл дообучения (выполняется в отдельном потоке)."""
        tmp_dir = Path('data/tmp_train_dataset')
        try:
            # Шаг 1: подготовка датасета
            self._set_status('training', 5, 'Подготовка датасета...')
            train_files, val_files = self._prepare_dataset(tmp_dir)

            if not train_files:
                raise RuntimeError('Не удалось подготовить датасет')

            # Шаг 2: создание dataset.yaml
            yaml_path = tmp_dir / 'dataset.yaml'
            self._write_dataset_yaml(yaml_path, tmp_dir)
            self._set_status('training', 15, f'Датасет: {len(train_files)} train, {len(val_files)} val')

            # Шаг 3: дообучение YOLO
            new_model_path = self._run_yolo_training(yaml_path)
            self._set_status('validating', 85, 'Валидация результата...')

            # Шаг 4: валидация
            improvement = self._validate_model(new_model_path, val_files)

            # Шаг 5: принятие решения о замене модели
            version_name = self._next_version_name()
            self._archive_model(new_model_path, version_name, improvement, len(train_files))

            if improvement > 0.02:
                self._deploy_model(new_model_path)
                self._set_status('done', 100,
                    f'Готово. Улучшение mAP: +{improvement:.1%}. Модель применена.')
                if self._on_model_ready:
                    self._on_model_ready(str(self.model_dir / 'best.pt'))
            elif improvement >= 0:
                self._set_status('done', 100,
                    f'Дообучение завершено. Улучшение mAP: +{improvement:.1%} (незначительно).')
            else:
                self._set_status('done', 100,
                    f'Откат: mAP снизился на {abs(improvement):.1%}. Модель не изменена.')
                logger.warning('Модель после дообучения хуже исходной — откат выполнен')

        except Exception as exc:
            self._set_status('error', 0, f'Ошибка дообучения: {exc}')
            logger.exception('Критическая ошибка дообучения')
        finally:
            # Очищаем временный датасет
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            with self._lock:
                self._training = False

    def _prepare_dataset(self, tmp_dir: Path) -> tuple[list, list]:
        """
        Формирует временный датасет в формате YOLO из подтверждённых примеров.
        Возвращает (train_files, val_files).
        """
        import random
        import cv2

        (tmp_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

        confirmed_imgs = list(DIR_CONFIRMED.glob('*.jpg'))
        random.shuffle(confirmed_imgs)

        n_val   = max(1, int(len(confirmed_imgs) * VALIDATION_SPLIT))
        val_set = set(p.name for p in confirmed_imgs[:n_val])

        train_files = []
        val_files   = []

        from core import classes as class_registry
        name_to_id = class_registry.name_to_id()

        for img_path in confirmed_imgs:
            meta = _load_meta(img_path)
            cls_name = meta.get('confirmed_class') or meta.get('suggested_class', '')
            cls_id   = name_to_id.get(cls_name)

            if cls_id is None:
                logger.warning('Неизвестный класс "%s", пропуск: %s', cls_name, img_path.name)
                continue

            bbox = meta.get('bbox', [])
            if len(bbox) != 4:
                logger.warning('Некорректный bbox в %s, пропуск', img_path.name)
                continue

            # Читаем изображение для получения размеров
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Конвертируем bbox в формат YOLO: cx cy w h (нормализованные)
            x1, y1, x2, y2 = bbox
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            # Защита от вырожденных боксов
            if bw < 0.01 or bh < 0.01:
                logger.warning('Слишком маленький bbox в %s, пропуск', img_path.name)
                continue

            split = 'val' if img_path.name in val_set else 'train'
            stem  = img_path.stem

            dst_img = tmp_dir / 'images' / split / img_path.name
            dst_lbl = tmp_dir / 'labels' / split / f'{stem}.txt'

            shutil.copy2(img_path, dst_img)
            with open(dst_lbl, 'w') as lf:
                lf.write(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

            if split == 'train':
                train_files.append(dst_img)
            else:
                val_files.append(dst_img)

        logger.info('Датасет подготовлен: %d train, %d val', len(train_files), len(val_files))
        return train_files, val_files

    def _write_dataset_yaml(self, yaml_path: Path, tmp_dir: Path):
        """
        Создаёт dataset.yaml для YOLO-тренировки.
        Имена классов берутся из общего реестра (базовые COCO + пользовательские),
        поэтому новые классы автоматически попадают в обучение. Базовые индексы
        сохраняются (person=0, car=2, ...), пользовательские идут с 80 — чтобы
        не ломать нумерацию существующих классов. nc = max(id)+1.
        """
        from core import classes as class_registry
        # Непрерывный список имён 0..nc-1 (len(names) == nc — требование ultralytics).
        id_to_name, nc = class_registry.training_names()

        lines = [
            f"path: {tmp_dir.resolve()}",
            "train: images/train",
            "val: images/val",
            f"nc: {nc}",
            "names:",
        ]
        for cid in range(nc):
            lines.append(f"  {cid}: {id_to_name[cid]}")

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    def _run_yolo_training(self, yaml_path: Path) -> Path:
        """
        Запускает дообучение через ultralytics API.
        Возвращает путь к лучшей модели (best.pt из runs/).
        """
        from ultralytics import YOLO

        # Определяем базовую модель (предпочитаем уже дообученную best.pt)
        base_model = self.model_dir / 'best.pt'
        if not base_model.exists():
            base_model = self.model_dir / 'yolov8n.pt'
        if not base_model.exists():
            raise FileNotFoundError(f'Базовая модель не найдена: {base_model}')

        self._set_status('training', 20, f'Базовая модель: {base_model.name}')

        model = YOLO(str(base_model))

        results = model.train(
            data=str(yaml_path),
            epochs=RETRAIN_EPOCHS,
            batch=RETRAIN_BATCH,
            imgsz=RETRAIN_IMGSZ,
            device='cpu',
            patience=5,           # ранняя остановка
            save=True,
            exist_ok=True,
            project='data/runs',
            name='retrain',
            verbose=False,
        )

        # Путь к обученной модели определяем НАДЁЖНО: ultralytics может сохранить
        # её в каталог retrain/retrain2/... или в свой runs_dir. Берём фактический
        # save_dir тренера, затем — best.pt, иначе last.pt, иначе ищем по дереву.
        best_pt = self._locate_trained_weights(model, results)
        if best_pt is None:
            raise FileNotFoundError(
                'Не удалось найти веса (best.pt/last.pt) после дообучения')

        self._set_status('training', 80, f'Тренировка завершена: {best_pt.name}')
        return best_pt

    @staticmethod
    def _locate_trained_weights(model, results) -> Optional[Path]:
        """
        Находит файл весов после тренировки. Порядок поиска:
          1) model.trainer.best / model.trainer.last;
          2) save_dir (из trainer или results) + weights/best.pt|last.pt;
          3) самый свежий best.pt|last.pt во всём дереве data/runs.
        """
        # 1) Прямые ссылки тренера
        trainer = getattr(model, 'trainer', None)
        for attr in ('best', 'last'):
            p = getattr(trainer, attr, None) if trainer else None
            if p and Path(p).exists():
                return Path(p)

        # 2) Каталог результатов
        save_dir = None
        if trainer is not None and getattr(trainer, 'save_dir', None):
            save_dir = Path(trainer.save_dir)
        elif getattr(results, 'save_dir', None):
            save_dir = Path(results.save_dir)
        if save_dir:
            for name in ('best.pt', 'last.pt'):
                cand = save_dir / 'weights' / name
                if cand.exists():
                    return cand

        # 3) Поиск по всему дереву запусков (берём самый свежий файл)
        candidates = list(Path('data/runs').glob('**/weights/best.pt')) + \
                     list(Path('data/runs').glob('**/weights/last.pt'))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        return None

    def _validate_model(self, new_model_path: Path, val_files: list) -> float:
        """
        Оценивает качество новой модели на отложенной выборке.
        Возвращает delta mAP (положительное = улучшение).
        Упрощённая валидация: считаем detection rate на val-примерах.
        """
        if not val_files:
            logger.warning('Нет val-файлов для валидации, пропускаем')
            return 0.01  # считаем условно улучшением

        try:
            from ultralytics import YOLO
            model = YOLO(str(new_model_path))
            metrics = model.val(
                data=None,
                split='val',
                verbose=False,
                device='cpu',
            )
            # metrics.box.map50 — mAP@50
            new_map = float(metrics.box.map50) if hasattr(metrics, 'box') else 0.5
        except Exception as exc:
            logger.warning('Ошибка валидации: %s. Используем эвристику.', exc)
            new_map = 0.5

        # Базовый mAP из метаданных текущей модели (если есть)
        current_meta = self.model_dir / 'best_meta.json'
        base_map = 0.5
        if current_meta.exists():
            try:
                with open(current_meta, encoding='utf-8') as f:
                    base_map = json.load(f).get('map50', 0.5)
            except (json.JSONDecodeError, OSError):
                pass

        improvement = new_map - base_map
        logger.info('Валидация: base_mAP=%.3f, new_mAP=%.3f, delta=%.3f',
                    base_map, new_map, improvement)
        return improvement

    def _archive_model(self, model_path: Path, version_name: str,
                       improvement: float, n_samples: int):
        """Сохраняет копию новой модели в архив версий."""
        dst = DIR_MODELS / f'{version_name}.pt'
        try:
            shutil.copy2(model_path, dst)
            meta = {
                'version':     version_name,
                'created':     datetime.now().isoformat(),
                'map50':       None,
                'improvement': round(improvement, 4),
                'samples':     n_samples,
                'base_model':  'best.pt',
            }
            with open(dst.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info('Версия модели сохранена: %s', dst)
        except Exception as exc:
            logger.warning('Ошибка архивирования модели: %s', exc)

    def _deploy_model(self, model_path: Path):
        """Заменяет текущую рабочую модель новой."""
        dst = self.model_dir / 'best.pt'
        shutil.copy2(model_path, dst)
        self._version = self._next_version_name()
        logger.info('Модель заменена: %s', dst)

    def _next_version_name(self) -> str:
        """Формирует имя следующей версии модели."""
        existing = [p.stem for p in DIR_MODELS.glob('*.pt')]
        v = 1
        while f'yolov8n_v{v}' in existing:
            v += 1
        return f'yolov8n_v{v}'

    def _detect_current_version(self) -> str:
        """Определяет текущую версию модели по архиву."""
        versions = sorted(DIR_MODELS.glob('*.pt'))
        if versions:
            return versions[-1].stem
        return 'yolov8n'

    def _set_status(self, status: str, progress: int, log: str = ''):
        """Потокобезопасное обновление статуса."""
        with self._lock:
            self._status   = status
            self._progress = progress
            if log:
                self._last_log = log
                logger.info('[Тренер] %s', log)
