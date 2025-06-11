import os
import sys
from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        # Avoid double run with autoreload
        if os.environ.get('RUN_MAIN') != 'true':
            return

        if len(sys.argv) > 1 and sys.argv[1] not in ['runserver', 'gunicorn', 'uvicorn']:
            return

        # Load NLP models only when actually running the server
        from api.services.nlp_manager import NLPManager
        NLPManager.get_instance().ensure_resources()