# Use Debian Bullseye (better for compiling dlib than slim)

FROM python:3.10-bullseye

# Set working directory

WORKDIR /app

# Install system dependencies required for dlib, opencv, and face-recognition

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    make \
    wget \
    curl \
    unzip \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libgtk2.0-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*


# Upgrade pip & setuptools

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python deps

COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy project files

COPY . .

# Expose port for Render

EXPOSE 8000

# Start FastAPI with uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
