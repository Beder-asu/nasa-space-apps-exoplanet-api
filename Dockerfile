FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY Procfile .
COPY railway.toml .

# Expose port
EXPOSE 8000

# Start command - use shell form to enable environment variable expansion
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}