"""
email_notify.py — отправка тревог по электронной почте (SMTP).

Зачем нужно (пункт ТЗ «настройка отправки тревог»):
  По сработавшему правилу система отправляет письмо на один или несколько
  адресов, при необходимости — со скриншотом тревоги во вложении.
  Параметры SMTP берутся из настроек (config.py -> email_*) и меняются
  через интерфейс без перезапуска сервера.

Транспорт выбирается по порту/флагу TLS:
  • порт 465                  → SMTP_SSL (неявный SSL, напр. Яндекс, Gmail SSL);
  • порт 587 и email_use_tls  → SMTP + STARTTLS;
  • иначе                     → обычный SMTP без шифрования.

Безопасность: пароль приложения SMTP в поставке пустой — оператор вводит
свой через интерфейс настроек.
"""

import logging
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

logger = logging.getLogger(__name__)

# Таймаут на сетевые операции SMTP, секунд
SMTP_TIMEOUT = 15


def _recipients(cfg: dict) -> list:
    """Список непустых адресов получателей."""
    return [a.strip() for a in (cfg.get('email_to') or []) if a and a.strip()]


def _build_message(cfg: dict, subject: str, body: str,
                   recipients: list, attachment: str | None) -> MIMEMultipart:
    """Собирает письмо. Если задан путь к картинке — прикладывает её."""
    sender = cfg.get('email_from') or cfg.get('email_login', '')

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From']    = sender
    msg['To']      = ', '.join(recipients)
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    if attachment and os.path.exists(attachment):
        try:
            with open(attachment, 'rb') as f:
                image = MIMEImage(f.read())
            image.add_header('Content-Disposition', 'attachment',
                             filename=os.path.basename(attachment))
            msg.attach(image)
        except OSError as exc:
            logger.warning('Не удалось приложить файл %s: %s', attachment, exc)

    return msg


def _open_server(host: str, port: int, use_tls: bool):
    """Открывает SMTP-соединение нужного типа (контекстный менеджер)."""
    if port == 465:
        context = ssl.create_default_context()
        return smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT, context=context)

    server = smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT)
    if use_tls:
        server.starttls(context=ssl.create_default_context())
    return server


def send_alert(cfg: dict, subject: str, body: str,
               attachment: str | None = None) -> tuple[bool, str]:
    """
    Отправляет письмо по настройкам cfg (словарь email_* из settings).
    Возвращает (успех, сообщение для интерфейса).
    """
    if not cfg.get('email_enabled'):
        return False, 'Отправка email отключена в настройках'

    recipients = _recipients(cfg)
    if not recipients:
        return False, 'Не задан ни один адрес получателя'

    host     = (cfg.get('email_smtp_host') or '').strip()
    port     = int(cfg.get('email_smtp_port', 587) or 587)
    login    = (cfg.get('email_login') or '').strip()
    password = cfg.get('email_password') or ''
    use_tls  = bool(cfg.get('email_use_tls', True))

    if not host or not login:
        return False, 'Не заданы параметры SMTP (хост/логин)'
    if not password:
        return False, 'Не задан пароль приложения SMTP'

    msg = _build_message(cfg, subject, body, recipients, attachment)

    try:
        with _open_server(host, port, use_tls) as server:
            server.login(login, password)
            server.send_message(msg)
        logger.info('Email отправлен: "%s" -> %s', subject, ', '.join(recipients))
        return True, f'Письмо отправлено ({len(recipients)} получателей)'
    except smtplib.SMTPAuthenticationError:
        logger.error('Ошибка аутентификации SMTP для %s', login)
        return False, 'Ошибка входа: проверьте логин и пароль приложения'
    except Exception as exc:
        logger.error('Ошибка отправки email: %s', exc)
        return False, f'Ошибка SMTP: {exc}'


def test_connection(cfg: dict) -> tuple[bool, str]:
    """Тестовое письмо для проверки настроек SMTP из интерфейса."""
    return send_alert(
        cfg,
        subject='Видеонаблюдение АЗС — проверка уведомлений',
        body=(
            'Это тестовое сообщение системы видеонаблюдения АЗС.\n'
            'Если письмо получено — параметры SMTP заданы верно, '
            'и тревоги по правилам будут приходить на эту почту.'
        ),
    )
