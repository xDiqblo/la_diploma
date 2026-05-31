"""
telegram_notify.py — отправка уведомлений в Telegram (текст + фото).

Чтобы получить token: написать @BotFather, создать бота, скопировать токен.
Чтобы узнать chat_id: написать своему боту любое сообщение и открыть
https://api.telegram.org/bot<TOKEN>/getUpdates — там будет "chat":{"id":...}.

Отправка выполняется в отдельном потоке, чтобы не тормозить детекцию.
"""
import threading
import requests

API_URL = "https://api.telegram.org/bot{token}/{method}"


def _post(token, method, data=None, files=None):
    """Низкоуровневый POST к Telegram API. Ошибки гасим, чтобы не падало приложение."""
    try:
        url = API_URL.format(token=token, method=method)
        resp = requests.post(url, data=data, files=files, timeout=10)
        return resp.ok
    except Exception as e:
        print(f"[Telegram] Ошибка отправки: {e}")
        return False


def send_message(token, chat_id, text):
    """Отправляет текстовое сообщение в фоне."""
    if not token or not chat_id:
        return
    threading.Thread(
        target=_post,
        args=(token, "sendMessage", {"chat_id": chat_id, "text": text}),
        daemon=True,
    ).start()


def send_photo(token, chat_id, photo_path, caption=""):
    """Отправляет фото (скриншот тревоги) с подписью в фоне."""
    if not token or not chat_id:
        return

    def _worker():
        try:
            with open(photo_path, "rb") as f:
                _post(token, "sendPhoto",
                      data={"chat_id": chat_id, "caption": caption},
                      files={"photo": f})
        except Exception as e:
            print(f"[Telegram] Не удалось отправить фото: {e}")

    threading.Thread(target=_worker, daemon=True).start()


def test_connection(token, chat_id):
    """Синхронная проверка из настроек: отправляет тестовое сообщение."""
    if not token or not chat_id:
        return False, "Не заданы token или chat_id"
    ok = _post(token, "sendMessage",
               {"chat_id": chat_id, "text": "✅ Тест связи: камера АЗС подключена к Telegram"})
    return ok, ("Сообщение отправлено" if ok else "Ошибка отправки (проверьте token/chat_id)")
