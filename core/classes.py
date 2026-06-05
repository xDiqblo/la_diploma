"""
classes.py — реестр классов объектов: базовые (COCO) + пользовательские.

Зачем нужно (пункт ТЗ «создание новых классов»):
  Оператор может добавить собственный класс («сигарета», «огонь», «дым»,
  «открытая дверь» и т.д.), задать ему имя и цвет рамки. Класс сохраняется
  в config/classes.json и сразу становится доступен:
    • при ручной разметке кадров (вкладка «Дообучение»);
    • при дообучении модели (попадает в dataset.yaml);
    • для подписи/раскраски рамок в детекторе.

Идентификаторы:
  Базовые классы используют их родные COCO class_id (person=0, car=2, ...).
  Пользовательские классы получают id начиная с CUSTOM_ID_START (80), чтобы
  не конфликтовать с 80 классами COCO. Это позволяет дообучать модель на
  новых классах, не ломая распознавание существующих.
"""

import json
import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

CLASSES_FILE    = 'config/classes.json'
CUSTOM_ID_START = 80

# Базовые классы COCO, которые система детектирует «из коробки».
# id, имя, цвет рамки (hex). Совпадает с SUPPORTED_CLASSES в detector.py.
BASE_CLASSES = [
    {'id': 0, 'name': 'person',     'color': '#00e676', 'custom': False},
    {'id': 1, 'name': 'bicycle',    'color': '#00bb88', 'custom': False},
    {'id': 2, 'name': 'car',        'color': '#4488ff', 'custom': False},
    {'id': 3, 'name': 'motorcycle', 'color': '#cc88ff', 'custom': False},
    {'id': 5, 'name': 'bus',        'color': '#ffaa44', 'custom': False},
    {'id': 7, 'name': 'truck',      'color': '#ff6020', 'custom': False},
]

# Полный список 80 классов COCO (индексы 0..79) — в точности соответствует
# предобученной модели yolov8n.pt. Нужен для формирования dataset.yaml при
# дообучении: чтобы сохранить уже известные классы и добавить пользовательские
# (с индекса 80), nc и длина names обязаны совпадать.
COCO80 = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
    43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
    58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
    63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush',
}

_lock = threading.Lock()


def _ensure_dir():
    os.makedirs(os.path.dirname(CLASSES_FILE), exist_ok=True)


def _read_custom() -> list:
    if not os.path.exists(CLASSES_FILE):
        return []
    try:
        with open(CLASSES_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.warning('classes.json повреждён, начинаем с пустого списка')
        return []


def _write_custom(items: list):
    _ensure_dir()
    with open(CLASSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def list_classes() -> list:
    """Полный список классов: базовые + пользовательские."""
    with _lock:
        return [dict(c) for c in BASE_CLASSES] + _read_custom()


def list_custom() -> list:
    with _lock:
        return _read_custom()


def _next_id(custom: list) -> int:
    used = {c['id'] for c in custom}
    cid = CUSTOM_ID_START
    while cid in used:
        cid += 1
    return cid


def add_class(name: str, color: str = '#a78bfa') -> dict:
    """
    Создаёт новый пользовательский класс.
    Возвращает {'ok': bool, 'error': str, 'class': dict}.
    """
    name = (name or '').strip()
    if not name:
        return {'ok': False, 'error': 'Имя класса не задано'}
    if len(name) > 40:
        return {'ok': False, 'error': 'Слишком длинное имя класса (макс. 40)'}
    # Разрешаем латиницу, кириллицу, цифры, пробел, дефис, подчёркивание
    if not re.fullmatch(r'[\wЀ-ӿ \-]+', name, flags=re.UNICODE):
        return {'ok': False, 'error': 'Имя содержит недопустимые символы'}

    if not re.fullmatch(r'#[0-9a-fA-F]{6}', color or ''):
        color = '#a78bfa'

    with _lock:
        custom = _read_custom()
        existing = {c['name'].lower() for c in BASE_CLASSES} | \
                   {c['name'].lower() for c in custom}
        if name.lower() in existing:
            return {'ok': False, 'error': f'Класс "{name}" уже существует'}

        new_cls = {'id': _next_id(custom), 'name': name,
                   'color': color, 'custom': True}
        custom.append(new_cls)
        _write_custom(custom)
    logger.info('Создан класс: %s (id=%d, цвет=%s)', name, new_cls['id'], color)
    return {'ok': True, 'class': new_cls}


def update_class(class_id: int, name: str = None, color: str = None) -> dict:
    """Изменяет имя/цвет пользовательского класса."""
    with _lock:
        custom = _read_custom()
        target = next((c for c in custom if c['id'] == class_id), None)
        if target is None:
            return {'ok': False, 'error': 'Класс не найден или базовый (не редактируется)'}
        if name:
            target['name'] = name.strip()[:40]
        if color and re.fullmatch(r'#[0-9a-fA-F]{6}', color):
            target['color'] = color
        _write_custom(custom)
    logger.info('Класс id=%d обновлён', class_id)
    return {'ok': True, 'class': target}


def delete_class(class_id: int) -> dict:
    """Удаляет пользовательский класс (базовые удалить нельзя)."""
    if class_id < CUSTOM_ID_START:
        return {'ok': False, 'error': 'Базовый класс нельзя удалить'}
    with _lock:
        custom = _read_custom()
        new_custom = [c for c in custom if c['id'] != class_id]
        if len(new_custom) == len(custom):
            return {'ok': False, 'error': 'Класс не найден'}
        _write_custom(new_custom)
    logger.info('Класс id=%d удалён', class_id)
    return {'ok': True}


def name_to_id() -> dict:
    """{имя_класса: id} для всех классов — используется при дообучении."""
    return {c['name']: c['id'] for c in list_classes()}


def color_map() -> dict:
    """{имя_класса: hex_цвет} — используется детектором для раскраски рамок."""
    return {c['name']: c['color'] for c in list_classes()}


def training_names() -> tuple[dict, int]:
    """
    Возвращает (names, nc) для dataset.yaml дообучения.

    names — непрерывный словарь индексов 0..nc-1 (без пропусков), иначе
    ultralytics падает с ошибкой «'names' length != 'nc'». Базовые 80 классов
    COCO сохраняются на своих местах (person=0, car=2, ...), пользовательские
    добавляются с индекса 80. Пропуски (если пользовательский id не подряд)
    заполняются заглушками, чтобы длина строго совпадала с nc.
    """
    names = dict(COCO80)
    for c in _read_custom():
        names[int(c['id'])] = c['name']
    nc = max(names) + 1
    for i in range(nc):
        names.setdefault(i, f'class_{i}')   # заполняем возможные пропуски
    return names, nc
