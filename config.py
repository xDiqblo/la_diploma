"""
config.py — загрузка и сохранение настроек приложения в JSON-файл.
"""
import json
import os
import threading

SETTINGS_FILE = 'data/settings.json'

DEFAULT_SETTINGS = {
    "confidence_threshold": 0.5,
    "frame_skip": 3,
    "active_classes": ["car", "bus", "truck"],
    "long_stay_seconds": 30,
    # Пустая строка = не подключаться при старте.
    # Оператор выбирает источник вручную на странице Монитора.
    "video_source": "",
    "alert_confidence": 0.7,
    "alert_on_appearance": False,
    "global_long_stay_alerts": False,
    "event_cooldown_seconds": 15,
    "save_screenshots": True,
    "save_video_clips": True,
    "clip_pre_seconds": 10,
    "clip_post_seconds": 10,
    "zones": [],
    "telegram_enabled": False,
    "telegram_token": "",
    "telegram_chat_id": "",
    "email_enabled": False,
    "email_recipients": [],
    "smtp_host": "",
    "smtp_port": 587,
    "smtp_user": "",
    "smtp_password": "",
    "email_event_types": ["zone", "anomaly"],
    "cloud_sync_enabled": False,
    "cloud_url": "http://localhost:8001",
    "cloud_api_key": "secret-key-123",
    "email_smtp_host": "",
    "email_smtp_port": 587,
    "email_use_tls": True,
    "email_login": "",
    "email_password": "",
    "email_from": "",
    "email_to": [],
    "alert_send_mode": "immediate",
    "alert_send_after_n": 2,
    "alert_send_delay": 5,
    "alert_attach_screenshot": True,
}

_lock = threading.Lock()


def load_settings() -> dict:
    with _lock:
        if not os.path.exists(SETTINGS_FILE):
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
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
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)