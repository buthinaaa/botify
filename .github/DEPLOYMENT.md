# Deployment Setup Guide

This guide explains how to set up automatic deployment for your Django app using GitHub Actions.

## Prerequisites

1. Your Django app is already deployed and running on the EC2 server
2. The project is cloned in `/home/admin/botify` on the server
3. A virtual environment is set up in `/home/admin/botify/env`
4. Nginx and Daphne are configured to serve the application

## Setup Instructions

### 1. Add SSH Key to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `EC2_SSH_KEY`
5. Value: Copy the entire content of your `Botify.pem` file (including the `-----BEGIN` and `-----END` lines)

### 2. Server Setup Requirements

Make sure your server has the following setup:

```bash
# Project should be in this location
/home/admin/botify/

# Virtual environment should exist
/home/admin/botify/env/

# Admin user should have sudo privileges for nginx
sudo systemctl reload nginx

# Git should be configured
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Nginx Configuration

Make sure your nginx is configured to proxy requests to Daphne running on port 8000.

Example nginx configuration (`/etc/nginx/sites-available/botify`):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /home/admin/botify/staticfiles/;
    }

    location /media/ {
        alias /home/admin/botify/media/;
    }
}
```

### 4. Service Management (Optional)

For better service management, you can create a systemd service for Daphne:

Create `/etc/systemd/system/daphne.service`:

```ini
[Unit]
Description=Daphne ASGI Server for Botify
After=network.target

[Service]
Type=simple
User=admin
Group=admin
WorkingDirectory=/home/admin/botify
Environment=PATH=/home/admin/botify/env/bin
ExecStart=/home/admin/botify/env/bin/daphne -b 0.0.0.0 -p 8000 config.asgi:application
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable daphne
sudo systemctl start daphne
```

If you use systemd service, update the deployment script to use:
```bash
sudo systemctl restart daphne
```
instead of the pkill/nohup commands.

## How the Deployment Works

1. **Trigger**: Deployment runs automatically when you push to the `main` branch
2. **Manual Deploy**: You can also trigger deployment manually from the GitHub Actions tab
3. **Process**:
   - **Testing Phase**:
     - Sets up Python environment
     - Installs dependencies
     - Runs flake8 linting to check code quality
     - Runs pytest tests in `api/tests/` directory
     - Only proceeds if all tests pass and linting is clean
   - **Deployment Phase** (only if tests pass):
     - Connects to your EC2 server via SSH
     - Pulls the latest code from the main branch
     - Installs/updates Python dependencies on server
     - Runs database migrations
     - Collects static files
     - Restarts the Daphne server
     - Reloads Nginx

## Configuration Files

The deployment workflow uses several configuration files:

- **`.flake8`**: Linting configuration for code quality checks
- **`pytest.ini`**: Test configuration (already exists in your project)
- **`.github/workflows/deploy.yml`**: The main deployment workflow

## Troubleshooting

### Check deployment logs:
- GitHub Actions logs: Go to your repo → Actions tab → Latest workflow run
- Server logs: 
  ```bash
  # Daphne logs
  tail -f /tmp/daphne.log
  
  # Nginx logs
  sudo tail -f /var/log/nginx/error.log
  sudo tail -f /var/log/nginx/access.log
  ```

### Common issues:

**Testing Phase:**
1. **Linting errors**: Fix flake8 violations in your code before pushing
2. **Test failures**: Check test logs in GitHub Actions and fix failing tests
3. **Missing dependencies**: Ensure all test dependencies are in `requirements.txt`

**Deployment Phase:**
1. **Permission denied**: Make sure the admin user has proper permissions
2. **Port already in use**: The pkill command should handle this, but you might need to manually kill processes
3. **Database migrations fail**: Check your database permissions and settings
4. **Static files not updating**: Clear browser cache or check nginx static file configuration
5. **Git pull errors**: Check if the repository is properly cloned on the server and has the correct remote origin

## Security Notes

- The SSH key is stored securely in GitHub Secrets
- The connection uses SSH key authentication
- The deployment script removes the SSH key after use
- All commands run with appropriate user permissions 