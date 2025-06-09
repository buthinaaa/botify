# Comprehensive Logging System for Chatbot Application

## Overview
I've added comprehensive logging throughout your chatbot application to track all operations, performance metrics, and errors. This will help you monitor, debug, and optimize your chatbot system.

## What's Been Added

### 1. Enhanced Message Views (`api/views/message_views.py`)
- **Request tracking**: Each request gets a unique ID for tracing
- **Database operations**: Logs all chatbot and message queries
- **WebSocket operations**: Tracks message sending to channels
- **Processing steps**: Detailed logging of each step in message generation
- **Performance timing**: Measures time for each operation
- **Error handling**: Comprehensive error logging with stack traces

### 2. Response Generation (`api/utilities/response_generation.py`)
- **Model API calls**: Logs all interactions with the LLM
- **Prompt building**: Tracks prompt construction process
- **Quality assessment**: Logs response quality evaluation
- **Fallback handling**: Tracks when and why fallbacks are triggered
- **Post-processing**: Logs response cleaning and enhancement

### 3. Intent Recognition (`api/utilities/intent_recognition.py`)
- **API calls**: Logs all zero-shot classification requests
- **Label matching**: Tracks which intents are detected
- **Fallback detection**: Logs when out-of-scope content is detected
- **Performance metrics**: Times each classification step

### 4. Message Processing (`api/utilities/message_processing.py`)
- **Text preprocessing**: Logs each step of message cleaning
- **Spell correction**: Tracks corrections made
- **Tokenization**: Logs token counts and processing
- **Lemmatization**: Times and tracks lemmatization

### 5. Sentiment Analysis (`api/utilities/sentiment_analysis.py`)
- **Model predictions**: Logs sentiment classifications
- **Context analysis**: Tracks when context is used for better prediction
- **Fallback handling**: Logs low-confidence scenarios
- **Trend analysis**: Tracks sentiment patterns over time

### 6. Named Entity Recognition (`api/utilities/ner.py`)
- **Entity extraction**: Logs all entities found
- **Confidence filtering**: Tracks which entities pass confidence thresholds
- **Performance timing**: Measures NER processing time

### 7. Retrieval System (`api/utilities/retrieval_system.py`)
- **Hybrid search**: Logs FAISS, BM25, and label similarity scores
- **Performance metrics**: Times each search component
- **Result ranking**: Logs top retrieved chunks with scores

## Log Files Generated

### 1. `general.log` (10MB, 5 backups)
- General application logs
- Django framework logs
- Overall system status

### 2. `chatbot.log` (50MB, 10 backups)
- Complete chatbot conversation flows
- Response generation processes
- Document retrieval operations
- Main chatbot functionality

### 3. `message_processing.log` (20MB, 7 backups)
- Text preprocessing steps
- Intent recognition
- Sentiment analysis
- NER processing
- All NLP pipeline operations

### 4. `performance.log` (20MB, 5 backups)
- Timing information for all operations
- Performance bottlenecks
- Model inference times
- Database query performance

### 5. `errors.log` (30MB, 10 backups)
- All errors and exceptions
- Stack traces
- Failed operations
- Error recovery attempts

## Setup Instructions

### 1. Add Logging Configuration to Django Settings
Copy the content from `logging_config_example.py` and add it to your `settings.py`:

```python
# At the top of settings.py
import os
from pathlib import Path

# Add the LOGGING configuration from logging_config_example.py
```

### 2. Create Logs Directory
The logging configuration will automatically create a `logs/` directory in your project root.

### 3. Environment Variables
Set the following environment variable for development:
```bash
export DJANGO_DEBUG=True  # For verbose logging in development
```

For production, either don't set this variable or set it to `False`.

## Log Levels Used

- **DEBUG**: Detailed information for debugging (parameter values, intermediate results)
- **INFO**: General information about program execution
- **WARNING**: Something unexpected happened but the program continues
- **ERROR**: A serious problem occurred

## Example Log Entries

### Request Processing
```
[INFO] 2024-01-15 14:30:25 | api.views.message_views | post:67 | [Request 140234567890] GenerateResponse.post started
[INFO] 2024-01-15 14:30:25 | api.views.message_views | post:70 | [Request 140234567890] Processing message: 'Hello, can you help me with pricing?' for chatbot: uuid-123
```

### Performance Tracking
```
[INFO] 2024-01-15 14:30:26 | api.utilities.response_generation | chatbot_response:450 | Message preprocessing completed in 0.045s
[INFO] 2024-01-15 14:30:26 | api.utilities.response_generation | chatbot_response:465 | Sentiment analysis completed in 0.120s
```

### Error Logging
```
[ERROR] 2024-01-15 14:30:30 | api.utilities.retrieval_system | hybrid_search:75 | FAISS search failed: index not found
[ERROR] 2024-01-15 14:30:30 | api.utilities.retrieval_system | hybrid_search:77 | Traceback (most recent call last): ...
```

## Monitoring and Analysis

### 1. Real-time Monitoring
```bash
# Watch all logs
tail -f logs/*.log

# Watch specific component
tail -f logs/chatbot.log

# Watch errors only
tail -f logs/errors.log
```

### 2. Performance Analysis
```bash
# Find slow operations
grep "completed in" logs/performance.log | sort -k9 -nr

# Check error patterns
grep -c "ERROR" logs/errors.log
```

### 3. Usage Patterns
```bash
# Count requests per hour
grep "GenerateResponse.post started" logs/chatbot.log | cut -d' ' -f2-3 | cut -d':' -f1-2 | sort | uniq -c

# Check most common intents
grep "Intent result:" logs/message_processing.log
```

## Benefits

1. **Debugging**: Trace issues through the entire request lifecycle
2. **Performance Optimization**: Identify bottlenecks and slow operations
3. **Error Monitoring**: Get notified of issues before users report them
4. **Usage Analytics**: Understand how your chatbot is being used
5. **Quality Assurance**: Monitor response quality and fallback rates
6. **Compliance**: Keep detailed records for audit purposes

## Log Rotation

The logging system uses rotating file handlers:
- Files are automatically rotated when they reach size limits
- Old logs are kept as backups (numbered .1, .2, etc.)
- Oldest logs are automatically deleted when backup limit is reached

## Production Considerations

1. **Disk Space**: Monitor log disk usage, especially `chatbot.log`
2. **Log Levels**: Consider reducing verbosity in production
3. **Sensitive Data**: Be careful not to log sensitive user information
4. **Performance**: Extensive logging may impact performance slightly
5. **Centralized Logging**: Consider using ELK stack or similar for production

## Customization

You can adjust the logging configuration by:
- Changing log levels for specific modules
- Modifying file sizes and backup counts
- Adding new log files for specific purposes
- Filtering out specific types of messages

The logging system is designed to be comprehensive yet flexible for your specific needs. 