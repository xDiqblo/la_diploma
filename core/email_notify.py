"""
email_notify.py — отправка тревог по электронной почте (SMTP).

Зачем нужно (пункт ТЗ «настройка отправки тревог»):
  По сработавшему правилу система отправляет письмо на один или несколько
  адресов. Поддерживается вложение скриншота. Параметры SMTP берутся из
  настроек (config.py -> email_*), меняются без перезапуска сервера.

Безопасность: пароль приложения SMTP в финальной поставке заменён на
заглушку — оператор вводит свой через интерфейс настроек.
"""

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def _build_message(cfg: dict, subject: str, body: str,
                   recipients: list, attachment: str = None) -> EmailMessage:
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From']    = cfg.get('email_from') or cfg.get('email_login', '')
    msg['To']      = ', '.join(recipients)
    msg.set_content(body)

    if attachment and os.path.exists(attachment):
        try:
            with open(attachment, 'rb') as f:
                data = f.read()
            msg.add_attachment(data, maintype='image', subtype='jpeg',
                               filename=os.path.basename(attachment))
        except OSError as exc:
            logger.warning('Не удалось приложить файл %s: %s', attachment, exc)
    return msg


def send_alert(cfg: dict, subject: str, body: str,
               attachment: str = None) -> tuple[bool, str]:
    """
    Отправляет письмо по настройкам cfg (словарь email_* из settings).
    Возвращает (успех, сообщение).
    """
    if not cfg.get('email_enabled'):
        return False, 'Отправка email отключена в настройках'

    recipients = [a.strip() for a in (cfg.get('email_to') or []) if a.strip()]
    if not recipients:
        return False, 'Не задан ни один адрес получателя'

    host = cfg.get('email_smtp_host', '')
    port = int(cfg.get('email_smtp_port', 587))
    login = cfg.get('email_login', '')
    password = cfg.get('email_password', '')
    if not host or not login:
        return False, 'Не заданы параметры SMTP (хост/логин)'

    if not cfg.get('email_from'):
        cfg = {**cfg, 'email_from': login}

    msg = _build_message(cfg, subject, body, recipients, attachment)

    try:
        if cfg.get('email_use_tls', True):
            context = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.starttls(context=context)
                if password:
                    server.login(login, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, port, timeout=15) as server:
                if password:
                    server.login(login, password)
                server.send_message(msg)
        logger.info('Email отправлен: "%s" -> %s', subject, ', '.join(recipients))
        return True, f'Письмо отправлено ({len(recipients)} получателей)'
    except Exception as exc:
        logger.error('Ошибка отправки email: %s', exc)
        return False, f'Ошибка SMTP: {exc}'


def test_connection(cfg: dict) -> tuple[bool, str]:
    """Тестовое письмо для проверки настроек SMTP."""
    return send_alert(
        cfg,
        subject='Тест: система видеонаблюдения АЗС',
        body='Проверка настроек отправки тревог. Если вы получили это письмо, '
             'параметры SMTP заданы верно.',
    )
