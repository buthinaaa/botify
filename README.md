# Botify - Dockerized Django Project

This project is a Django application that uses PostgreSQL with pgvector extension for vector similarity search capabilities.

## Prerequisites

- Docker (version 20.10.0 or higher)
- Docker Compose (version 2.0.0 or higher)
- Git (optional, for cloning the repository)

## Project Structure

```
botify/
├── api/                 # API endpoints
├── config/             # Django project configuration
├── staticfiles/        # Static files
├── .env                # Environment variables
├── requirements.txt    # Python dependencies
├── manage.py          # Django management script
├── Dockerfile         # Docker configuration
└── docker-compose.yml # Docker Compose configuration
```

## Getting Started

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd botify
   ```

2. **Create environment variables**:
   - Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
   - Update the `.env` file with your desired configuration

3. **Build and start the containers**:
   ```bash
   docker-compose up --build
   ```
   This will:
   - Build the Django application container
   - Start the PostgreSQL database with pgvector
   - Run the migration
   - Run the Django development server

4. **Create a superuser** (optional):
   ```bash
   docker-compose exec api python manage.py createsuperuser
   ```

## Accessing the Application

- **Django Application**: http://localhost:8000
- **Admin Interface**: http://localhost:8000/admin
- **API Endpoints**: http://localhost:8000/api/

## Database Configuration

The PostgreSQL database is configured with the following defaults:
- Host: db
- Port: 5432
- Database: botify
- User: postgres
- Password: postgres

## Development Workflow

1. **Making changes to the code**:
   - The application code is mounted as a volume, so changes will be reflected immediately
   - No need to rebuild the container for code changes

2. **Installing new dependencies**:
   - Add new packages to `requirements.txt`
   - Rebuild the containers:
   ```bash
   docker-compose up --build
   ```

3. **Viewing logs**:
   ```bash
   docker-compose logs -f
   ```

## Common Commands

- **Stop the containers**:
  ```bash
  docker-compose down
  ```

- **Stop and remove volumes** (including database data):
  ```bash
  docker-compose down -v
  ```

- **Run management commands**:
  ```bash
  docker-compose exec api python manage.py <command>
  ```

- **Access the database**:
  ```bash
  docker-compose exec db psql -U postgres -d botify
  ```

## Troubleshooting

1. **Port conflicts**:
   - If port 8000 or 5432 is already in use, modify the ports in `docker-compose.yml`

2. **Database issues**:
   - If the database isn't starting properly, try:
   ```bash
   docker-compose down -v
   docker-compose up --build
   ```

3. **Container issues**:
   - To rebuild a specific service:
   ```bash
   docker-compose up --build api
   ```

## Production Deployment

For production deployment, you should:
1. Set appropriate environment variables
2. Use a production-grade WSGI server (Gunicorn is already configured)
3. Configure proper security settings
4. Set up proper static file serving
5. Use a production-grade database backup strategy