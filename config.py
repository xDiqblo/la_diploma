"""
config.py — загрузка и сохранение настроек приложения в JSON-файл.
Все настройки меняются через веб-интерфейс без перезапуска сервера.
"""
import json
import os
import threading

SETTINGS_FILE = 'data/settings.json'

DEFAULT_SETTINGS = {
    # ── детекция ──────────────────────────────────────────────────────────────
    "confidence_threshold": 0.5,     # порог уверенности YOLO (0.1–0.9)
    "frame_skip": 3,                  # YOLO раз в N кадров (выше = быстрее)
    # Активные классы. Вот здесь — правильное решение проблемы «машины как люди»:
    # просто снимаем галочку с person. Когда студент дообучит модель — включит обратно.
    "active_classes": ["car", "bus", "truck"],
    "long_stay_seconds": 30,          # порог аномалии «долгое нахождение»
    # ── источник видео ────────────────────────────────────────────────────────
    "video_source": "test_videos/test_video3.mp4",              # "0" — USB-камера, или RTSP/путь к файлу
    # ── тревоги и сохранение ─────────────────────────────────────────────────
    # ВАЖНО (исправление «слишком много событий»):
    #   Раньше тревога создавалась на КАЖДЫЙ новый объект в кадре — проезжающие
    #   машины засыпали папки скриншотами и видео без всякого смысла.
    #   Теперь это поведение по умолчанию ВЫКЛЮЧЕНО. Тревоги создаются только:
    #     1) по правилам ЗОН (см. "zones" ниже) — основной механизм;
    #     2) опционально — глобальное «долгое нахождение» во всём кадре.
    "alert_confidence": 0.7,
    "alert_on_appearance": False,     # тревога на любое появление объекта (шумно — выкл)
    "global_long_stay_alerts": False, # тревога на долгое нахождение во всём кадре
    "event_cooldown_seconds": 15,     # антидребезг: мин. пауза между однотипными тревогами
    "save_screenshots": True,
    "save_video_clips": True,
    "clip_pre_seconds": 10,
    "clip_post_seconds": 10,
    # ── зоны срабатывания ─────────────────────────────────────────────────────
    # Список зон. Каждая зона — многоугольник в относительных координатах (0..1),
    # чтобы не зависеть от разрешения камеры. У каждой зоны свои правила, и по
    # каждой считаются ОТДЕЛЬНЫЕ метрики/отчётность. Зоны рисуются на плеере
    # (страница «Монитор»). Подробная структура зоны описана в core/zones.py.
    "zones": [],
    # ── Telegram ──────────────────────────────────────────────────────────────
    "telegram_enabled": False,
    "telegram_token": "",
    "telegram_chat_id": "",
    # ── облачная синхронизация ────────────────────────────────────────────────
    "cloud_sync_enabled": False,
    "cloud_url": "http://localhost:8001",
    "cloud_api_key": "secret-key-123",
    # ── отправка тревог по email (см. core/email_notify.py и core/rules.py) ────
    # Хранится здесь, чтобы меняться без перезапуска сервера. Пароль приложения
    # SMTP в финальной поставке заменён на заглушку.
    "email_enabled": False,
    "email_smtp_host": "smtp.gmail.com",
    "email_smtp_port": 587,
    "email_use_tls": True,
    "email_login": "",
    "email_password": "",
    "email_from": "",
    "email_to": [],                   # список адресов получателей
    "alert_send_mode": "immediate",   # immediate | after_n | delay
    "alert_send_after_n": 2,          # для after_n: сколько срабатываний накопить
    "alert_send_delay": 5,            # для delay: задержка в секундах
    "alert_attach_screenshot": True,  # прикладывать скриншот к письму
}

# Кастомные классы, правила тревог и зоны вынесены в отдельные файлы
# (core/classes.py, core/rules.py, core/zone_store.py), потому что у них своя
# логика импорта/экспорта и они меняются независимо от общих настроек.

_lock = threading.Lock()


def load_settings() -> dict:
    """Читает настройки из файла. При отсутствии файла создаёт с дефолтами."""
    with _lock:
        if not os.path.exists(SETTINGS_FILE):
            _write(DEFAULT_SETTINGS)
            return dict(DEFAULT_SETTINGS)
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            merged = dict(DEFAULT_SETTINGS)
            merged.update(data)
            return merged
        except (json.JSONDecodeError, OSError):
            return dict(DEFAULT_SETTINGS)


def save_settings(new_settings: dict) -> dict:
    """Сохраняет обновлённые настройки (принимает частичный словарь)."""
    with _lock:
        current = dict(DEFAULT_SETTINGS)
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    current.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        current.update(new_settings)
        _write(current)
        return current


def _write(settings: dict):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
