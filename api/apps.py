from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        import os
        # Avoid running twice in dev server
        if os.environ.get('RUN_MAIN', None) != 'true':
            from api.services.nlp_manager import NLPManager
            NLPManager.get_instance().ensure_resources()