"""
detector.py — детектор объектов (YOLOv8n + ByteTrack).

Решение проблемы «машины определяются как люди»:
  НЕ делаем переименование class_id=0 -> 'car' — это костыль.
  Правильное решение: оператор сам выбирает в настройках, какие классы
  детектировать. На АЗС с наклонной камерой нужно включить только
  car/bus/truck (снять галочку с person). Когда студент дообучит модель —
  просто добавит person обратно. Архитектура не сломается.

Оптимизации производительности:
  - imgsz=416 (вместо 1280) — в ~3x быстрее на CPU
  - frame_skip: YOLO запускается раз в N кадров, рамки рисуются каждый кадр
    по закешированным координатам — нет мерцания, FPS выше
"""

import cv2
import time
from ultralytics import YOLO

# Все поддерживаемые COCO-классы с цветами рамок (BGR)
# Ключ — COCO class_id, значение — (внутреннее_имя, цвет_BGR)
SUPPORTED_CLASSES = {
    0: ('person',      (0, 220, 80)),    # зелёный
    2: ('car',         (220, 80, 0)),    # синий
    5: ('bus',         (0, 210, 210)),   # жёлтый
    7: ('truck',       (0, 140, 255)),   # оранжевый
    3: ('motorcycle',  (180, 0, 240)),   # фиолетовый
    1: ('bicycle',     (0, 180, 120)),   # зелёный тёмный
}

# Обратный маппинг: имя → class_id (нужен для фильтрации по именам)
NAME_TO_ID = {name: cid for cid, (name, _) in SUPPORTED_CLASSES.items()}


class Detector:
    def __init__(
        self,
        model_path='yolov8n.pt',
        confidence_threshold=0.5,
        frame_skip=3,
        active_classes=None,     # список имён: ['car', 'bus', 'truck']
        long_stay_seconds=30,
        imgsz=416,
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.frame_skip = max(1, int(frame_skip))
        self.long_stay_seconds = long_stay_seconds
        self.imgsz = imgsz
        self.enabled = False
        self.frame_counter = 0

        # Время первого появления каждого track_id (для подсчёта времени в кадре)
        self.track_start_time = {}

        # Кеш последних результатов — отдаём на пропущенных кадрах (нет мерцания)
        self.last_detections = []
        self.last_class_counts = {}
        self.last_draw_items = []  # список (bbox, color, label) для отрисовки

        # Применяем начальный список классов
        if active_classes is None:
            active_classes = ['car', 'bus', 'truck']
        self._apply_active_classes(active_classes)

    # ─── управление классами ─────────────────────────────────────────────────

    def _apply_active_classes(self, class_names: list):
        """Вычисляет множество активных COCO class_id по списку имён."""
        self.active_class_ids = {
            NAME_TO_ID[n] for n in class_names if n in NAME_TO_ID
        }
        self.active_class_names = set(class_names)

    # ─── публичный интерфейс управления ─────────────────────────────────────

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self.track_start_time.clear()
            self.last_detections = []
            self.last_class_counts = {}
            self.last_draw_items = []

    def update_settings(
        self,
        confidence_threshold=None,
        frame_skip=None,
        active_classes=None,
        long_stay_seconds=None,
    ):
        """Применяет новые настройки «на лету» без перезапуска сервера."""
        if confidence_threshold is not None:
            self.confidence_threshold = float(confidence_threshold)
        if frame_skip is not None:
            self.frame_skip = max(1, int(frame_skip))
        if active_classes is not None:
            self._apply_active_classes(active_classes)
        if long_stay_seconds is not None:
            self.long_stay_seconds = float(long_stay_seconds)

    # ─── основная обработка кадра ────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Возвращает (кадр_с_рамками, список_детекций, счётчики_по_классам).
        YOLO запускается раз в frame_skip кадров — на остальных кадрах
        переиспользуем последние результаты (нет мерцания, выше FPS).
        """
        if not self.enabled:
            return frame, [], {}

        self.frame_counter += 1
        if self.frame_counter % self.frame_skip == 0:
            self._run_yolo(frame)

        self._draw_boxes(frame)
        return frame, self.last_detections, self.last_class_counts

    def _run_yolo(self, frame):
        """Запускает YOLO+ByteTrack и обновляет кеш."""
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            classes=list(self.active_class_ids),  # YOLO фильтрует классы до инференса
            tracker='bytetrack.yaml',
        )

        detections = []
        draw_items = []
        counts = {}
        now = time.time()
        active_ids = set()

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            data = boxes.data.cpu().numpy()

            for det in data:
                x1, y1, x2, y2 = map(int, det[:4])
                confidence = float(det[4])
                class_id = int(det[5])

                if class_id not in SUPPORTED_CLASSES:
                    continue

                class_name, base_color = SUPPORTED_CLASSES[class_id]

                # track_id от ByteTrack (может отсутствовать в первых кадрах)
                track_id = None
                if boxes.id is not None and len(det) > 6:
                    track_id = int(det[6])
                    active_ids.add(track_id)

                # Время нахождения объекта в кадре
                elapsed = 0.0
                if track_id is not None:
                    self.track_start_time.setdefault(track_id, now)
                    elapsed = now - self.track_start_time[track_id]

                is_long_stay = elapsed >= self.long_stay_seconds
                counts[class_name] = counts.get(class_name, 0) + 1

                detections.append({
                    'class':      class_name,
                    'confidence': round(confidence, 2),
                    'track_id':   track_id,
                    'time':       round(elapsed, 1),
                    'bbox':       [x1, y1, x2, y2],
                    'long_stay':  is_long_stay,
                })

                # Красная рамка при долгом нахождении
                color = (0, 0, 200) if is_long_stay else base_color
                id_str = f'ID:{track_id}' if track_id is not None else 'ID:?'
                label = f'{class_name} {id_str} {elapsed:.0f}s {confidence:.2f}'
                draw_items.append(((x1, y1, x2, y2), color, label))

        # Убираем из истории track_id, вышедшие из кадра
        for tid in list(self.track_start_time):
            if tid not in active_ids:
                del self.track_start_time[tid]

        self.last_detections = detections
        self.last_class_counts = counts
        self.last_draw_items = draw_items

    def _draw_boxes(self, frame):
        """Рисует рамки по кешированным координатам (вызывается на каждом кадре)."""
        for (x1, y1, x2, y2), color, label in self.last_draw_items:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Тёмный фон под текст — читаемо на любом фоне
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame,
                          (x1, max(0, y1 - th - 8)),
                          (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(frame, label,
                        (x1 + 2, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
