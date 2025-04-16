#!/bin/bash

# Run migrations
python manage.py migrate

# Execute the command passed to the container
exec "$@" 