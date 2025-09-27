
FROM python:3.10-slim

WORKDIR /app

# Install system packages required for dlib & opencv
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

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies, force prebuilt binaries
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy project files
COPY . .

EXPOSE 8000

# Start FastAPI with uvicorn (adjust if your entrypoint is different)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

