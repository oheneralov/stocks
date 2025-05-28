# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt if you have one
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install tensorflow keras && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy application code
COPY . .

# Set default command
CMD ["python", "main.py"]