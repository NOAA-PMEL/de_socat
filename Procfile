workers: celery -A app:celery_app worker --concurrency=4
worker-beat: celery -A app:celery_app beat
web: gunicorn app:server --workers 4