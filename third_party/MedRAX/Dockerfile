# MedRAX Docker Image
# Medical Reasoning Agent for Chest X-ray

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdcm-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/temp /app/logs /model-weights && \
    chmod -R 777 /app/temp /app/logs

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install the package in editable mode
RUN pip install -e .

# Expose Gradio port
EXPOSE 8585

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8585/ || exit 1

# Run the application
CMD ["python3", "main.py"]
