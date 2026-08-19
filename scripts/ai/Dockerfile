FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

WORKDIR /app

# Install git and utilities
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install required python packages
RUN pip install numpy nibabel scipy

# Clone and install nnInteractive
RUN git clone https://github.com/MIC-DKFZ/nnInteractive.git /opt/nnInteractive && \
    cd /opt/nnInteractive && \
    pip install -e .

# Copy everything from scripts/ai into the container
COPY . /app

# Pre-download nnInteractive models
RUN python -c "import sys; sys.path.insert(0, '/app/helpnet_bundle'); import nnInteractive.model_management as mm; mm.ensure_model_available('nnInteractive_v1.0')"

# Expose port for JSON/TCP communication
EXPOSE 5005

# Run the python worker (unbuffered so logs are visible)
CMD ["python", "-u", "python_worker.py"]
