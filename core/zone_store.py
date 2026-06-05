"""
zone_store.py — хранение зон в отдельных файлах zones/zoneX.json.

Зачем нужно (пункт ТЗ «настройка зон»):
  Каждая зона сохраняется в отдельный файл zones/zoneX.json (X — номер зоны).
  Это позволяет импортировать/экспортировать зоны по одной и переносить их
  между установками. Источник правды по зонам — этот модуль (а не settings.json).

Важное требование ТЗ:
  При перезапуске сервера все зоны загружаются ВЫКЛЮЧЕННЫМИ (enabled=False),
  но существующими. Оператор включает нужные зоны вручную. Реализовано в
  load_all(reset_enabled=True), который вызывается один раз при старте.
"""

import json
import logging
import os
import re
import threading
import time

logger = logging.getLogger(__name__)

ZONES_DIR = 'zones'
_lock = threading.Lock()


def _ensure_dir():
    os.makedirs(ZONES_DIR, exist_ok=True)


def _zone_files() -> list:
    if not os.path.isdir(ZONES_DIR):
        return []
    files = [f for f in os.listdir(ZONES_DIR)
             if re.fullmatch(r'zone\d+\.json', f)]
    # сортировка по числовому индексу, а не лексикографически
    return sorted(files, key=lambda f: int(re.search(r'\d+', f).group()))


def _normalize(z: dict) -> dict:
    """Дефолты для зоны (совпадают с ZoneManager._normalize)."""
    return {
        'id':               str(z.get('id') or f'z{int(time.time()*1000)}'),
        'name':             z.get('name') or 'Зона',
        'points':           z.get('points') or [],
        'color':            z.get('color') or '#00c8ff',
        'enabled':          bool(z.get('enabled', False)),
        'rule':             z.get('rule') or 'presence',
        'classes':          z.get('classes') or [],
        'min_confidence':   float(z.get('min_confidence', 0.5)),
        'dwell_seconds':    float(z.get('dwell_seconds', 20)),
        'max_count':        int(z.get('max_count', 3)),
        'cooldown_seconds': float(z.get('cooldown_seconds', 30)),
        'save_media':       bool(z.get('save_media', True)),
        'rule_ids':         z.get('rule_ids') or [],   # привязанные правила тревог
    }


def load_all(reset_enabled: bool = False) -> list:
    """
    Читает все зоны из zones/zoneX.json.
    reset_enabled=True — принудительно выключает все зоны (вызов при старте).
    """
    with _lock:
        zones = []
        for fname in _zone_files():
            try:
                with open(os.path.join(ZONES_DIR, fname), encoding='utf-8') as f:
                    z = _normalize(json.load(f))
                if reset_enabled:
                    z['enabled'] = False
                zones.append(z)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning('Зона %s не прочитана: %s', fname, exc)
        if reset_enabled and zones:
            # сохраняем выключенное состояние на диск
            _write_all_locked(zones)
            logger.info('Загружено зон: %d (все выключены при старте)', len(zones))
        return zones


def _write_all_locked(zones: list):
    """Перезаписывает файлы зон (вызывать под _lock)."""
    _ensure_dir()
    # удаляем старые файлы зон
    for fname in _zone_files():
        try:
            os.remove(os.path.join(ZONES_DIR, fname))
        except OSError:
            pass
    # пишем zone1.json .. zoneN.json в порядке списка
    for i, z in enumerate(zones, start=1):
        path = os.path.join(ZONES_DIR, f'zone{i}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_normalize(z), f, ensure_ascii=False, indent=2)


def save_all(zones: list) -> list:
    """Сохраняет полный список зон (каждую в свой zoneX.json)."""
    norm = [_normalize(z) for z in zones if z.get('points')]
    with _lock:
        _write_all_locked(norm)
    logger.info('Сохранено зон: %d', len(norm))
    return norm


def export_zone(zone_id: str) -> dict | None:
    """Возвращает зону по id (для экспорта в файл)."""
    for z in load_all():
        if z['id'] == zone_id:
            return z
    return None


def import_zone(data: dict) -> dict:
    """Добавляет зону из импортированного JSON. Возвращает {'ok', 'zone'|'error'}."""
    if not data.get('points'):
        return {'ok': False, 'error': 'В импортируемой зоне нет точек'}
    zones = load_all()
    z = _normalize(data)
    z['enabled'] = False          # импортированная зона выключена по умолчанию
    z['id'] = f'z{int(time.time()*1000)}'  # новый id, чтобы не было коллизий
    zones.append(z)
    save_all(zones)
    return {'ok': True, 'zone': z}
