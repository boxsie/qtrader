# QTrader Data Service Dockerfile
# Lightweight container for data fetching, news, and LLM analysis

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache optimization)
COPY dataservice/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared event models
COPY shared /app/shared

# Copy dataservice code
COPY dataservice /app/dataservice

WORKDIR /app/dataservice

# Create directories for volume mounts
RUN mkdir -p /app/data /app/storage

# Expose port
EXPOSE 8766

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8766/health')"

# Run server
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8766"]
