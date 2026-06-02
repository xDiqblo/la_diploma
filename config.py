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
    "video_source": "0",              # "0" — USB-камера, или RTSP/путь к файлу
    # ── тревоги и сохранение ─────────────────────────────────────────────────
    "alert_confidence": 0.7,
    "save_screenshots": True,
    "save_video_clips": True,
    "clip_pre_seconds": 10,
    "clip_post_seconds": 10,
    # ── Telegram ──────────────────────────────────────────────────────────────
    "telegram_enabled": False,
    "telegram_token": "",
    "telegram_chat_id": "",
    # ── облачная синхронизация ────────────────────────────────────────────────
    "cloud_sync_enabled": False,
    "cloud_url": "http://localhost:8001",
    "cloud_api_key": "secret-key-123",
}

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
