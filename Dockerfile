FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model artifacts
COPY src/ src/
COPY models/ models/

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
