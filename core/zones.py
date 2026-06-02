"""
zones.py — «зоны срабатывания» на видеопотоке.

Зачем нужно:
  Оператор рисует на плеере один или несколько многоугольников (зон). Для каждой
  зоны задаётся СВОЁ правило тревоги, и по каждой зоне считаются ОТДЕЛЬНЫЕ метрики.
  Это решает две задачи диплома сразу:
    • убирает «мусорные» события — тревога возникает только в нужной области кадра
      (например, у конкретной колонки), а не на каждую проезжающую машину;
    • даёт отдельную отчётность: сколько объектов было в каждой зоне, сколько
      тревог она дала и т.д.

Структура одной зоны (хранится в settings.json -> "zones"):
  {
    "id":            "z1",                 # уникальный идентификатор
    "name":          "Колонка 1",          # человекочитаемое имя (попадает в отчёты)
    "points":        [[x,y], ...],         # вершины в ОТНОСИТЕЛЬНЫХ координатах 0..1
    "color":         "#00c8ff",            # цвет рамки зоны (hex)
    "enabled":       true,                 # учитывать ли зону
    "rule":          "long_stay",          # presence | long_stay | count
    "classes":       [],                   # какие классы считать ([] = все активные)
    "min_confidence":0.5,                  # порог уверенности для зоны
    "dwell_seconds": 20,                   # для rule=long_stay: сколько ждать
    "max_count":     3,                    # для rule=count: порог скопления
    "cooldown_seconds": 30,                # антидребезг между тревогами зоны
    "save_media":    true                  # сохранять ли скриншот/видео по тревоге
  }

Координаты относительные (0..1), поэтому зоны не «съезжают» при смене разрешения.
Точкой объекта считаем НИЗ рамки (середина нижней грани) — это место, где объект
«стоит на земле», так попадание в зону определяется корректнее, чем по центру.
"""
import time

import cv2
import numpy as np

# Человекочитаемые названия типов событий по зонам (попадают в БД и отчёты)
EVENT_PRESENCE  = 'Зона: вход'
EVENT_LONG_STAY = 'Зона: задержка'
EVENT_COUNT     = 'Зона: скопление'


def _hex_to_bgr(hex_color: str):
    """'#00c8ff' -> (255, 200, 0) в формате BGR для OpenCV."""
    try:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)
    except (ValueError, IndexError):
        return (255, 200, 0)


def _point_in_polygon(px, py, poly) -> bool:
    """
    Классический алгоритм «трассировки луча» (ray casting).
    Пускаем горизонтальный луч вправо от точки и считаем пересечения со сторонами
    многоугольника: нечётное число пересечений → точка внутри.
    poly — список вершин [(x,y), ...] в пикселях.
    """
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


class ZoneManager:
    """Хранит зоны, считает метрики и порождает тревоги по правилам зон."""

    def __init__(self, zones=None, cooldown_default=30):
        self.cooldown_default = cooldown_default
        self._zones = []
        self._state = {}          # zone_id -> состояние (счётчики, таймеры)
        self.update_zones(zones or [])

    # ─── управление списком зон ──────────────────────────────────────────────

    def update_zones(self, zones: list):
        """Заменяет список зон. Состояние удалённых зон забываем, новых — заводим."""
        self._zones = [self._normalize(z) for z in zones if z.get('points')]
        live_ids = {z['id'] for z in self._zones}

        # выкидываем состояние исчезнувших зон
        for zid in list(self._state):
            if zid not in live_ids:
                del self._state[zid]

        # заводим состояние для новых зон
        for z in self._zones:
            self._state.setdefault(z['id'], {
                'count':        0,     # объектов в зоне прямо сейчас
                'total_events': 0,     # всего тревог зоны за сессию
                'visitors':     set(), # уникальные track_id, побывавшие в зоне
                'enter_time':   {},    # track_id -> время входа (для задержки)
                'alerted':      set(), # track_id, по которым задержка уже сработала
                'last_event_t': 0.0,   # время последней тревоги (антидребезг)
            })

    @staticmethod
    def _normalize(z: dict) -> dict:
        """Подставляет значения по умолчанию для полей зоны."""
        return {
            'id':             str(z.get('id') or f"z{int(time.time()*1000)}"),
            'name':           z.get('name') or 'Зона',
            'points':         z.get('points') or [],
            'color':          z.get('color') or '#00c8ff',
            'enabled':        bool(z.get('enabled', True)),
            'rule':           z.get('rule') or 'presence',
            'classes':        z.get('classes') or [],
            'min_confidence': float(z.get('min_confidence', 0.5)),
            'dwell_seconds':  float(z.get('dwell_seconds', 20)),
            'max_count':      int(z.get('max_count', 3)),
            'cooldown_seconds': float(z.get('cooldown_seconds', 30)),
            'save_media':     bool(z.get('save_media', True)),
        }

    def reset_runtime(self):
        """Сброс «живых» счётчиков и таймеров (при старте/остановке детекции)."""
        for st in self._state.values():
            st['count'] = 0
            st['enter_time'].clear()
            st['alerted'].clear()
            st['last_event_t'] = 0.0

    # ─── основная обработка кадра ─────────────────────────────────────────────

    def process(self, detections, frame_w, frame_h, now=None):
        """
        Для текущего кадра пересчитывает попадание объектов в зоны и формирует
        тревоги по правилам. Возвращает список словарей-тревог:
            {'zone': имя, 'event_type': тип, 'obj': детекция, 'save_media': bool}
        Метрики по зонам обновляются здесь же (см. get_stats / get_live).
        """
        if now is None:
            now = time.time()
        if frame_w <= 0 or frame_h <= 0:
            return []

        triggered = []

        for z in self._zones:
            st = self._state[z['id']]
            if not z['enabled']:
                st['count'] = 0
                continue

            poly = [(p[0] * frame_w, p[1] * frame_h) for p in z['points']]
            allowed = set(z['classes'])           # пусто = принимаем все классы

            inside_ids = set()
            inside_objs = []
            for obj in detections:
                if obj.get('confidence', 0) < z['min_confidence']:
                    continue
                if allowed and obj.get('class') not in allowed:
                    continue
                x1, y1, x2, y2 = obj['bbox']
                foot_x, foot_y = (x1 + x2) / 2.0, float(y2)   # «ноги» объекта
                if _point_in_polygon(foot_x, foot_y, poly):
                    inside_objs.append(obj)
                    tid = obj.get('track_id')
                    if tid is not None:
                        inside_ids.add(tid)

            st['count'] = len(inside_objs)
            st['visitors'].update(inside_ids)

            # обновляем тайминги входа/выхода для правила «задержка»
            for tid in inside_ids:
                st['enter_time'].setdefault(tid, now)
            for tid in list(st['enter_time']):
                if tid not in inside_ids:          # объект покинул зону — забываем
                    st['enter_time'].pop(tid, None)
                    st['alerted'].discard(tid)

            ev = self._apply_rule(z, st, inside_objs, inside_ids, now)
            if ev:
                triggered.append(ev)

        return triggered

    def _apply_rule(self, z, st, inside_objs, inside_ids, now):
        """Проверяет правило конкретной зоны и при срабатывании возвращает тревогу."""
        rule = z['rule']
        cooldown = z['cooldown_seconds']

        def fire(event_type, obj):
            st['last_event_t'] = now
            st['total_events'] += 1
            return {
                'zone':       z['name'],
                'event_type': event_type,
                'obj':        obj,
                'save_media': z['save_media'],
            }

        # presence — тревога при появлении объекта в зоне (с антидребезгом)
        if rule == 'presence':
            if inside_objs and (now - st['last_event_t']) >= cooldown:
                return fire(EVENT_PRESENCE, inside_objs[0])

        # long_stay — объект пробыл в зоне дольше dwell_seconds
        elif rule == 'long_stay':
            for obj in inside_objs:
                tid = obj.get('track_id')
                if tid is None or tid in st['alerted']:
                    continue
                if now - st['enter_time'].get(tid, now) >= z['dwell_seconds']:
                    st['alerted'].add(tid)        # по этому объекту больше не дублируем
                    return fire(EVENT_LONG_STAY, obj)

        # count — в зоне одновременно слишком много объектов
        elif rule == 'count':
            if len(inside_objs) >= z['max_count'] and \
               (now - st['last_event_t']) >= cooldown:
                return fire(EVENT_COUNT, inside_objs[0])

        return None

    # ─── метрики и отрисовка ─────────────────────────────────────────────────

    def get_live(self) -> list:
        """Живые метрики по зонам (для панели на странице «Монитор»)."""
        out = []
        for z in self._zones:
            st = self._state[z['id']]
            out.append({
                'id':           z['id'],
                'name':         z['name'],
                'color':        z['color'],
                'rule':         z['rule'],
                'enabled':      z['enabled'],
                'count':        st['count'],
                'total_events': st['total_events'],
                'visitors':     len(st['visitors']),
            })
        return out

    def draw(self, frame):
        """Рисует зоны поверх кадра: полупрозрачная заливка + имя + счётчик."""
        if not self._zones:
            return frame
        h, w = frame.shape[:2]
        overlay = frame.copy()

        for z in self._zones:
            if len(z['points']) < 3:
                continue
            color = _hex_to_bgr(z['color'])
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in z['points']],
                           dtype=np.int32)
            active = z['enabled']

            # полупрозрачная заливка только у активных зон
            if active:
                cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], True, color, 2 if active else 1)

            # подпись: имя зоны + сколько объектов внутри
            st = self._state[z['id']]
            label = f"{z['name']} [{st['count']}]"
            lx, ly = int(pts[:, 0].min()), int(pts[:, 1].min())
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (lx, max(0, ly - th - 8)),
                          (lx + tw + 6, ly), color, -1)
            cv2.putText(frame, label, (lx + 3, max(12, ly - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # смешиваем заливку с кадром (прозрачность 25%)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        return frame
