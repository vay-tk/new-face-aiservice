# Use Python 3.10 slim

FROM python:3.10-slim

# Set working directory

WORKDIR /app

# Install required system dependencies for dlib & opencv

RUN apt-get update && apt-get install -y 
cmake 
g++ 
make 
wget 
curl 
unzip 
pkg-config 
libopenblas-dev 
liblapack-dev 
libgtk2.0-dev 
libglib2.0-0 
libsm6 
libxext6 
libxrender-dev 
python3-dev 
&& rm -rf /var/lib/apt/lists/*

# Upgrade pip and tools

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency file

COPY requirements.txt .

# Install Python dependencies (force binary if available)

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy project files

COPY . .

# Expose port

EXPOSE 8000

# Run the FastAPI app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
