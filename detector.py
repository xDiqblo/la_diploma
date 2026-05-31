import cv2
import time
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, frame_skip=2):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.enabled = False
        self.frame_skip = frame_skip
        self.frame_counter = 0
        self.track_history = {}
        self.track_start_time = {}
        self.last_detections = []
        self.last_class_counts = {}

    def set_enabled(self, enabled):
        self.enabled = enabled
        if not enabled:
            self.track_start_time.clear()

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold

    def process_frame(self, frame):
        if not self.enabled:
            return frame, [], {}

        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return frame, self.last_detections, self.last_class_counts

        # ✅ ВАЖНО: используем .track() для получения ID
        results = self.model.track(frame,
                                   persist=True,
                                   verbose=False,
                                   conf=self.confidence_threshold,
                                   imgsz=640)  # можно вернуть 640 для скорости

        detections_info = []
        class_counts = {'person': 0, 'car': 0, 'bus': 0, 'truck': 0}
        current_time = time.time()

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.data.cpu().numpy()

            # Проверяем, есть ли ID у объектов
            has_ids = results[0].boxes.id is not None

            for detection in boxes:
                x1, y1, x2, y2 = map(int, detection[:4])
                confidence = float(detection[4])
                class_id = int(detection[5])

                # Фильтр классов (person, car, bus, truck)
                if class_id not in {0, 2, 5, 7}:
                    continue

                class_names = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck'}
                class_name = class_names[class_id]

                colors = {0: (0, 255, 0), 2: (255, 0, 0), 5: (0, 255, 255), 7: (0, 165, 255)}
                color = colors.get(class_id, (255, 255, 255))

                # Получаем track_id (если есть)
                track_id = None
                if has_ids and len(detection) > 6:
                    track_id = int(detection[6])

                # Время нахождения объекта в кадре
                elapsed_time = 0
                if track_id is not None:
                    if track_id not in self.track_start_time:
                        self.track_start_time[track_id] = current_time
                    elapsed_time = current_time - self.track_start_time[track_id]

                # Счётчики
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                # Данные для веб-интерфейса
                detections_info.append({
                    'class': class_name,
                    'confidence': round(confidence, 2),
                    'track_id': track_id,
                    'time': round(elapsed_time, 1),
                    'bbox': [x1, y1, x2, y2]
                })

                # Рисуем рамку
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Текст на рамке
                id_text = f"ID:{track_id}" if track_id is not None else "ID:—"
                label = f"{class_name} {id_text} {elapsed_time:.1f}s {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, detections_info, class_counts