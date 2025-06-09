# Add this to your Django settings.py file for comprehensive logging

import os
from pathlib import Path

# Define base directory for logs
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / 'logs'

# Create logs directory if it doesn't exist
LOG_DIR.mkdir(exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
        'detailed': {
            'format': '[{levelname}] {asctime} | {name} | {funcName}:{lineno} | {message}',
            'style': '{',
        },
        'request_format': {
            'format': '[{levelname}] {asctime} | Request {message}',
            'style': '{',
        }
    },
    'handlers': {
        # Console handler for development
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'detailed'
        },
        
        # File handler for general application logs
        'file_general': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOG_DIR / 'general.log',
            'maxBytes': 1024*1024*10,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed'
        },
        
        # File handler for chatbot-specific logs
        'file_chatbot': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOG_DIR / 'chatbot.log',
            'maxBytes': 1024*1024*50,  # 50MB
            'backupCount': 10,
            'formatter': 'detailed'
        },
        
        # File handler for message processing logs
        'file_message_processing': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOG_DIR / 'message_processing.log',
            'maxBytes': 1024*1024*20,  # 20MB
            'backupCount': 7,
            'formatter': 'detailed'
        },
        
        # File handler for performance/timing logs
        'file_performance': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOG_DIR / 'performance.log',
            'maxBytes': 1024*1024*20,  # 20MB
            'backupCount': 5,
            'formatter': 'detailed'
        },
        
        # File handler for errors only
        'file_errors': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOG_DIR / 'errors.log',
            'maxBytes': 1024*1024*30,  # 30MB
            'backupCount': 10,
            'formatter': 'detailed'
        }
    },
    'loggers': {
        # Root logger
        '': {
            'handlers': ['console', 'file_general', 'file_errors'],
            'level': 'INFO',
            'propagate': False,
        },
        
        # Django's logger
        'django': {
            'handlers': ['console', 'file_general'],
            'level': 'INFO',
            'propagate': False,
        },
        
        # Message views logger
        'api.views.message_views': {
            'handlers': ['console', 'file_chatbot', 'file_errors'],
            'level': 'DEBUG',
            'propagate': False,
        },
        
        # Response generation logger
        'api.utilities.response_generation': {
            'handlers': ['console', 'file_chatbot', 'file_performance', 'file_errors'],
            'level': 'DEBUG',
            'propagate': False,
        },
        
        # Intent recognition logger
        'api.utilities.intent_recognition': {
            'handlers': ['console', 'file_message_processing', 'file_errors'],
            'level': 'DEBUG',
            'propagate': False,
        },
        
        # Message processing logger
        'api.utilities.message_processing': {
            'handlers': ['console', 'file_message_processing', 'file_errors'],
            'level': 'DEBUG',
            'propagate': False,
        },
        
        # Sentiment analysis logger
        'api.utilities.sentiment_analysis': {
            'handlers': ['console', 'file_message_processing', 'file_errors'],
            'level': 'DEBUG',
            'propagate': False,
        },
        
        # NER logger
        'api.utilities.ner': {
            'handlers': ['console', 'file_message_processing', 'file_errors'],
            'level': 'DEBUG',
            'propagate': False,
        },
        
        # Retrieval system logger
        'api.utilities.retrieval_system': {
            'handlers': ['console', 'file_chatbot', 'file_performance', 'file_errors'],
            'level': 'DEBUG',
            'propagate': False,
        },
        
        # NLP Manager logger (if you want to log that too)
        'api.services.nlp_manager': {
            'handlers': ['console', 'file_performance', 'file_errors'],
            'level': 'INFO',
            'propagate': False,
        }
    }
}

# Optional: Set different log levels for development vs production
if os.environ.get('DJANGO_DEBUG', 'False').lower() == 'true':
    # Development settings - more verbose logging
    LOGGING['handlers']['console']['level'] = 'DEBUG'
    LOGGING['loggers']['']['level'] = 'DEBUG'
else:
    # Production settings - less verbose console logging
    LOGGING['handlers']['console']['level'] = 'WARNING'
    LOGGING['loggers']['']['level'] = 'INFO' 