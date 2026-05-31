import cv2
import time
from ultralytics import YOLO


class Detector:
    """
    Детектор объектов на базе YOLOv8n + трекинг ByteTrack.

    Главные улучшения по сравнению с первой версией:
    1. Маппинг person(id=0) -> car: камера АЗС стоит под наклоном, и YOLO
       часто принимает машины за людей. Поэтому при включённом флаге
       treat_person_as_car все объекты класса person считаются автомобилями.
    2. Детекция (тяжёлый YOLO) выполняется раз в frame_skip кадров, а ОТРИСОВКА
       рамок делается на КАЖДОМ кадре по последним известным координатам.
       Это убирает мерцание рамок и поднимает FPS.
    """

    # Классы COCO, которые нас интересуют: 0-person, 2-car, 5-bus, 7-truck
    CLASS_NAMES = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'}
    # Цвета рамок (BGR) для каждого класса
    CLASS_COLORS = {'person': (0, 255, 0), 'car': (255, 0, 0),
                    'bus': (0, 255, 255), 'truck': (0, 165, 255)}

    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5,
                 frame_skip=3, treat_person_as_car=True, long_stay_seconds=30,
                 imgsz=480):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.frame_skip = max(1, frame_skip)      # минимум 1 (каждый кадр)
        self.treat_person_as_car = treat_person_as_car
        self.long_stay_seconds = long_stay_seconds
        self.imgsz = imgsz                          # меньше размер -> быстрее на CPU

        self.enabled = False
        self.frame_counter = 0

        # Время первого появления каждого track_id (для подсчёта «времени в кадре»)
        self.track_start_time = {}

        # Последние результаты детекции — отдаём их и на пропущенных кадрах,
        # чтобы интерфейс и рамки не «прыгали».
        self.last_detections = []
        self.last_class_counts = self._empty_counts()
        # Готовые элементы для отрисовки: (bbox, color, label) — рисуем каждый кадр
        self.last_draw_items = []

    # ---------- Вспомогательные методы ----------
    @staticmethod
    def _empty_counts():
        return {'person': 0, 'car': 0, 'bus': 0, 'truck': 0}

    def set_enabled(self, enabled):
        self.enabled = enabled
        if not enabled:
            # При остановке сбрасываем состояние, чтобы счётчики обнулились
            self.track_start_time.clear()
            self.last_detections = []
            self.last_class_counts = self._empty_counts()
            self.last_draw_items = []

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold

    def update_settings(self, confidence_threshold=None, frame_skip=None,
                        treat_person_as_car=None, long_stay_seconds=None):
        """Применяет настройки «на лету» (вызывается со страницы «Настройки»)."""
        if confidence_threshold is not None:
            self.confidence_threshold = float(confidence_threshold)
        if frame_skip is not None:
            self.frame_skip = max(1, int(frame_skip))
        if treat_person_as_car is not None:
            self.treat_person_as_car = bool(treat_person_as_car)
        if long_stay_seconds is not None:
            self.long_stay_seconds = float(long_stay_seconds)

    # ---------- Основной метод обработки кадра ----------
    def process_frame(self, frame):
        """
        Возвращает (frame_с_рамками, список_детекций, счётчики_по_классам).
        Тяжёлый YOLO запускается раз в frame_skip кадров, отрисовка — каждый кадр.
        """
        if not self.enabled:
            return frame, [], self._empty_counts()

        self.frame_counter += 1
        run_detection = (self.frame_counter % self.frame_skip == 0)

        # Запускаем YOLO только на «рабочих» кадрах
        if run_detection:
            self._run_detection(frame)

        # Рамки рисуем ВСЕГДА по последним известным данным (нет мерцания)
        self._draw_last_items(frame)

        return frame, self.last_detections, self.last_class_counts

    def _run_detection(self, frame):
        """Запуск YOLO + трекинг, обновление self.last_* данных."""
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            tracker="bytetrack.yaml",
        )

        detections_info = []
        draw_items = []
        class_counts = self._empty_counts()
        current_time = time.time()
        active_ids = set()

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            data = boxes.data.cpu().numpy()
            has_ids = boxes.id is not None

            for det in data:
                x1, y1, x2, y2 = map(int, det[:4])
                confidence = float(det[4])
                class_id = int(det[5])

                # Фильтр: только интересующие классы
                if class_id not in self.CLASS_NAMES:
                    continue

                class_name = self.CLASS_NAMES[class_id]

                # ГЛАВНЫЙ ФИКС: на АЗС людей почти нет, а машины часто
                # детектируются как person -> переименовываем person в car.
                if self.treat_person_as_car and class_name == 'person':
                    class_name = 'car'

                color = self.CLASS_COLORS[class_name]

                # track_id (если трекер его присвоил)
                track_id = None
                if has_ids and len(det) > 6:
                    track_id = int(det[6])
                    active_ids.add(track_id)

                # Время нахождения объекта в кадре
                elapsed_time = 0.0
                if track_id is not None:
                    if track_id not in self.track_start_time:
                        self.track_start_time[track_id] = current_time
                    elapsed_time = current_time - self.track_start_time[track_id]

                # Флаг аномалии «долгое нахождение»
                is_long_stay = elapsed_time >= self.long_stay_seconds

                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                detections_info.append({
                    'class': class_name,
                    'confidence': round(confidence, 2),
                    'track_id': track_id,
                    'time': round(elapsed_time, 1),
                    'bbox': [x1, y1, x2, y2],
                    'long_stay': is_long_stay,
                })

                # Долгое нахождение подсвечиваем красным
                box_color = (0, 0, 255) if is_long_stay else color
                id_text = f"ID:{track_id}" if track_id is not None else "ID:-"
                label = f"{class_name} {id_text} {elapsed_time:.0f}s {confidence:.2f}"
                draw_items.append(((x1, y1, x2, y2), box_color, label))

        # Чистим историю времени для объектов, ушедших из кадра
        if active_ids:
            for tid in list(self.track_start_time.keys()):
                if tid not in active_ids:
                    del self.track_start_time[tid]

        self.last_detections = detections_info
        self.last_class_counts = class_counts
        self.last_draw_items = draw_items

    def _draw_last_items(self, frame):
        """Рисует рамки и подписи по сохранённым данным последней детекции."""
        for (x1, y1, x2, y2), color, label in self.last_draw_items:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
