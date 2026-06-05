"""
rules.py — пользовательские правила тревог.

Зачем нужно (пункт ТЗ «новые правила для отправки тревог»):
  После того как модель дообучена на новом классе (например, «сигарета»),
  оператор создаёт правило вида «Курение на территории АЗС»:
      ЕСЛИ класс = "сигарета" [И зона = "АЗС-восток"] ТО тревога.
  Правила хранятся в config/rules.json, могут включаться/выключаться,
  привязываться к зоне и иметь временные ограничения и антидребезг.

Структура правила:
  {
    "id":            "r1",
    "name":          "Курение на территории АЗС",
    "enabled":       true,
    "classes":       ["сигарета", "огонь"],   # сработает по любому из классов
    "zone_id":       "z123" | null,            # null = весь кадр
    "min_count":     1,                         # минимум объектов одновременно
    "cooldown_seconds": 30,                     # пауза между тревогами правила
    "time_from":     "00:00",                   # окно активности (по часам суток)
    "time_to":       "23:59"
  }

Оценка правил вынесена из main_web, чтобы логика тревог была в одном месте
и легко тестировалась отдельно.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

RULES_FILE = 'config/rules.json'

_lock = threading.Lock()
# Время последнего срабатывания каждого правила (антидребезг). В памяти —
# при перезапуске сбрасывается, что корректно.
_last_fired: dict[str, float] = {}


def _ensure_dir():
    os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)


def _read() -> list:
    if not os.path.exists(RULES_FILE):
        return []
    try:
        with open(RULES_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.warning('rules.json повреждён, начинаем с пустого списка')
        return []


def _write(items: list):
    _ensure_dir()
    with open(RULES_FILE, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _normalize(r: dict) -> dict:
    return {
        'id':               str(r.get('id') or f'r{int(time.time()*1000)}'),
        'name':             (r.get('name') or 'Правило').strip(),
        'enabled':          bool(r.get('enabled', True)),
        'classes':          [str(c) for c in (r.get('classes') or [])],
        'zone_id':          r.get('zone_id') or None,
        'min_count':        max(1, int(r.get('min_count', 1))),
        'cooldown_seconds': float(r.get('cooldown_seconds', 30)),
        'time_from':        r.get('time_from') or '00:00',
        'time_to':          r.get('time_to') or '23:59',
    }


def list_rules() -> list:
    with _lock:
        return _read()


def save_rule(rule: dict) -> dict:
    """Создаёт или обновляет правило (по id)."""
    norm = _normalize(rule)
    with _lock:
        items = _read()
        idx = next((i for i, r in enumerate(items) if r['id'] == norm['id']), None)
        if idx is None:
            items.append(norm)
        else:
            items[idx] = norm
        _write(items)
    logger.info('Правило сохранено: %s (%s)', norm['name'], norm['id'])
    return {'ok': True, 'rule': norm}


def delete_rule(rule_id: str) -> dict:
    with _lock:
        items = _read()
        new_items = [r for r in items if r['id'] != rule_id]
        if len(new_items) == len(items):
            return {'ok': False, 'error': 'Правило не найдено'}
        _write(new_items)
    _last_fired.pop(rule_id, None)
    logger.info('Правило удалено: %s', rule_id)
    return {'ok': True}


def rules_for_zone(zone_id: str) -> list:
    """Правила, привязанные к конкретной зоне (для контекстного меню зоны)."""
    return [r for r in list_rules() if r.get('zone_id') == zone_id]


def _within_time(rule: dict, now_dt: datetime) -> bool:
    """Проверяет, попадает ли текущее время в окно активности правила."""
    try:
        cur = now_dt.strftime('%H:%M')
        tf, tt = rule['time_from'], rule['time_to']
        if tf <= tt:
            return tf <= cur <= tt
        # окно через полночь (например, 22:00–06:00)
        return cur >= tf or cur <= tt
    except Exception:
        return True


def evaluate(zone_membership: dict, class_counts: dict, now: float = None) -> list:
    """
    Оценивает все включённые правила.

    zone_membership: {zone_id: {class_name: count}} — что внутри каждой зоны.
    class_counts:    {class_name: count} — всего в кадре (для правил без зоны).

    Возвращает список сработавших правил:
        [{'rule_id', 'name', 'zone_id', 'classes', 'matched_class', 'count'}]
    """
    if now is None:
        now = time.time()
    now_dt = datetime.fromtimestamp(now)
    fired = []

    for rule in list_rules():
        if not rule.get('enabled'):
            continue
        if not _within_time(rule, now_dt):
            continue

        # Источник подсчёта: зона или весь кадр
        if rule.get('zone_id'):
            counts = zone_membership.get(rule['zone_id'], {})
        else:
            counts = class_counts or {}

        # Срабатывание по любому из перечисленных классов
        matched_class = None
        matched_count = 0
        for cls in rule['classes']:
            c = counts.get(cls, 0)
            if c >= rule['min_count'] and c > matched_count:
                matched_class = cls
                matched_count = c

        if matched_class is None:
            continue

        # Антидребезг
        if now - _last_fired.get(rule['id'], 0) < rule['cooldown_seconds']:
            continue
        _last_fired[rule['id']] = now

        fired.append({
            'rule_id':       rule['id'],
            'name':          rule['name'],
            'zone_id':       rule.get('zone_id'),
            'classes':       rule['classes'],
            'matched_class': matched_class,
            'count':         matched_count,
        })
        logger.info('Правило сработало: %s (класс=%s, кол-во=%d)',
                    rule['name'], matched_class, matched_count)

    return fired


def reset_runtime():
    """Сброс антидребезга (при старте/остановке детекции)."""
    _last_fired.clear()
