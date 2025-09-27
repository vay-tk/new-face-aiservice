```dockerfile
# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for dlib, face-recognition, and opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    make \
    libopenblas-dev \
    liblapack-dev \
    libgtk2.0-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port Render will use
EXPOSE 8000

# Run your app (adjust if you use uvicorn/flask/django)
# Example: Flask app in app.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

```
