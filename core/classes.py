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
