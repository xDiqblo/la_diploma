"""
config.py — загрузка и сохранение настроек приложения в JSON-файл.
Настройки можно менять через веб-страницу «Настройки» без перезапуска сервера.
"""
import json
import os
import threading

# Путь к файлу настроек
SETTINGS_FILE = 'settings.json'

# Настройки по умолчанию (используются при первом запуске)
DEFAULT_SETTINGS = {
    "confidence_threshold": 0.5,   # порог уверенности YOLO (0.1–0.9)
    "frame_skip": 3,               # запускать YOLO раз в N кадров (больше = выше FPS)
    "treat_person_as_car": True,   # считать класс person (id=0) автомобилем (ракурс АЗС)
    "long_stay_seconds": 30,       # порог «долгого нахождения» объекта (секунд)
    "alert_confidence": 0.7,       # порог уверенности для сохранения тревоги/скриншота
    "save_screenshots": True,      # сохранять скриншоты при тревоге
    "save_video_clips": True,      # сохранять видеофрагменты при тревоге
    "clip_pre_seconds": 10,        # сколько секунд видео ДО тревоги сохранять
    "clip_post_seconds": 10,       # сколько секунд видео ПОСЛЕ тревоги сохранять
    "telegram_enabled": False,     # включить Telegram-уведомления
    "telegram_token": "",          # токен бота от @BotFather
    "telegram_chat_id": "",        # ID чата для отправки уведомлений
}

# Блокировка на случай одновременного чтения/записи из разных потоков
_lock = threading.Lock()


def load_settings():
    """Читает настройки из файла. Если файла нет — создаёт со значениями по умолчанию."""
    with _lock:
        if not os.path.exists(SETTINGS_FILE):
            _save_unlocked(DEFAULT_SETTINGS)
            return dict(DEFAULT_SETTINGS)
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Добавляем недостающие ключи (на случай обновления версии)
            settings = dict(DEFAULT_SETTINGS)
            settings.update(data)
            return settings
        except (json.JSONDecodeError, OSError):
            # Файл повреждён — возвращаем значения по умолчанию
            return dict(DEFAULT_SETTINGS)


def save_settings(new_settings):
    """Сохраняет настройки в файл. Принимает словарь (можно частичный)."""
    with _lock:
        current = dict(DEFAULT_SETTINGS)
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    current.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        current.update(new_settings)
        _save_unlocked(current)
        return current


def _save_unlocked(settings):
    """Запись в файл без блокировки (вызывается внутри методов, где блокировка уже взята)."""
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
