# Dockerfile (simple, production-friendly)
FROM python:3.10-slim

# Keep Python from writing .pyc files and make logs appear immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install any small system deps your Python packages need (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Create a non-root user for safety and switch to it (optional but recommended)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Expose the port the app will run on
EXPOSE 8000

# Default command: uvicorn development server (replace with Gunicorn for production)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
