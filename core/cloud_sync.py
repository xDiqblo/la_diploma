"""
cloud_sync.py — фоновая синхронизация событий с облачным сервером.

Логика:
  - Фоновый поток раз в 30 секунд смотрит события с synced=0 в локальной БД.
  - Для каждого: POST /api/events на облако + отдельный запрос со скриншотом.
  - При успехе — ставим synced=1.
  - При недоступности облака — накапливаются в SQLite, уйдут при восстановлении.
"""
import os
import threading

import requests

import database

SYNC_INTERVAL = 30  # секунд между попытками синхронизации


class CloudSync:
    def __init__(self, cloud_url: str, api_key: str, enabled: bool = False):
        self.cloud_url = cloud_url.rstrip('/')
        self.api_key = api_key
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='cloud-sync'
        )
        self._thread.start()
        print('[CloudSync] Фоновая синхронизация запущена')

    def stop(self):
        self._stop.set()

    def update(self, cloud_url: str, api_key: str, enabled: bool):
        """Применяет новые настройки на лету."""
        self.cloud_url = cloud_url.rstrip('/')
        self.api_key = api_key
        self.enabled = enabled

    def _loop(self):
        while not self._stop.is_set():
            if self.enabled:
                try:
                    self._sync_batch()
                except Exception as e:
                    print(f'[CloudSync] Ошибка: {e}')
            self._stop.wait(SYNC_INTERVAL)

    def _sync_batch(self):
        """Отправляет пачку несинхронизированных событий."""
        events = database.get_unsynced(limit=20)
        if not events:
            return

        headers = {'X-API-Key': self.api_key}

        for ev in events:
            try:
                # Метаданные события
                payload = {
                    'id':         ev['id'],
                    'timestamp':  ev['timestamp'],
                    'event_type': ev['event_type'],
                    'class_name': ev['class_name'],
                    'track_id':   ev['track_id'],
                    'confidence': ev['confidence'],
                }
                resp = requests.post(
                    f'{self.cloud_url}/api/events',
                    json=payload, headers=headers, timeout=5,
                )
                if resp.status_code not in (200, 201):
                    print(f'[CloudSync] Событие {ev["id"]} отклонено: {resp.status_code}')
                    continue

                database.mark_synced(ev['id'])

                # Скриншот отправляем отдельным запросом
                shot = ev.get('screenshot')
                if shot and os.path.exists(shot):
                    with open(shot, 'rb') as f:
                        requests.post(
                            f'{self.cloud_url}/api/events/{ev["id"]}/screenshot',
                            files={'file': f},
                            headers=headers,
                            timeout=10,
                        )

            except requests.exceptions.ConnectionError:
                print('[CloudSync] Облако недоступно, следующая попытка через 30 сек')
                break  # не теряем время на остальные, облако недоступно
            except Exception as e:
                print(f'[CloudSync] Ошибка события {ev["id"]}: {e}')
