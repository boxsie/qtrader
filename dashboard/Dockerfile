# QTrader Dashboard Server Dockerfile
# Lightweight container for dashboard web server (no GPU needed)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared event models
COPY shared /app/shared

# Copy dashboard server files
COPY dashboard/server.py .
COPY dashboard/index.html .

# Expose dashboard port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8765/health')"

# Run server (bind to 0.0.0.0 to accept external connections in Docker)
CMD ["python", "server.py", "--host", "0.0.0.0"]
