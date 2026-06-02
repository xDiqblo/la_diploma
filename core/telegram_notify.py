"""
telegram_notify.py — уведомления в Telegram (текст + фото).

Как настроить:
  1. Пишем @BotFather → /newbot → копируем токен.
  2. Пишем своему боту любое сообщение, открываем:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     Ищем "chat": {"id": ...} — это chat_id.

Все отправки — в фоновом потоке, чтобы не блокировать детекцию.
"""
import threading
import requests

_API = 'https://api.telegram.org/bot{token}/{method}'


def _post(token: str, method: str, data=None, files=None) -> bool:
    """POST к Telegram API. Ошибки подавляем — не критично для работы системы."""
    try:
        url = _API.format(token=token, method=method)
        resp = requests.post(url, data=data, files=files, timeout=10)
        return resp.ok
    except Exception as e:
        print(f'[Telegram] Ошибка: {e}')
        return False


def send_message(token: str, chat_id: str, text: str):
    """Отправляет текстовое сообщение (в фоне)."""
    if not token or not chat_id:
        return
    threading.Thread(
        target=_post,
        args=(token, 'sendMessage', {'chat_id': chat_id, 'text': text}),
        daemon=True,
    ).start()


def send_photo(token: str, chat_id: str, photo_path: str, caption: str = ''):
    """Отправляет фото со скриншотом тревоги (в фоне)."""
    if not token or not chat_id:
        return

    def _worker():
        try:
            with open(photo_path, 'rb') as f:
                _post(token, 'sendPhoto',
                      data={'chat_id': chat_id, 'caption': caption},
                      files={'photo': f})
        except Exception as e:
            print(f'[Telegram] Ошибка отправки фото: {e}')

    threading.Thread(target=_worker, daemon=True).start()


def test_connection(token: str, chat_id: str) -> tuple[bool, str]:
    """Синхронная проверка: шлёт тестовое сообщение, возвращает (ok, текст)."""
    if not token or not chat_id:
        return False, 'Не заданы token или chat_id'
    ok = _post(token, 'sendMessage', {
        'chat_id': chat_id,
        'text': '✅ Тест связи: камера АЗС подключена к Telegram'
    })
    return ok, ('Сообщение доставлено' if ok else 'Ошибка (проверьте token / chat_id)')
