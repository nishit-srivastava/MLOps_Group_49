# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional, minimal here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose Flask default port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
