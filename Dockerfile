# Use official slim Python base image
FROM python:3.10-slim

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv $VIRTUAL_ENV

# Copy the repo code into container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e ".[webui]"

# Download model checkpoints
RUN mkdir -p /app/checkpoints && \
    wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth -P /app/checkpoints && \
    wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth -P /app/checkpoints && \
    wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model -P /app/checkpoints && \
    wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth -P /app/checkpoints && \
    wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth -P /app/checkpoints && \
    wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab -P /app/checkpoints

# Expose port for Cloud Run
EXPOSE 8080

# Start the web UI app using gunicorn
CMD exec gunicorn webui:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4
