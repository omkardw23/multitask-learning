# Use NVIDIA CUDA PyTorch base image for GPU acceleration (CU12 for compatibility with your dependencies)
FROM nvidia/cuda:12.4.127-runtime-ubuntu20.04

# Set up Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory inside the container
WORKDIR /app

# Copy requirements first for caching (Speeds up builds)
COPY requirements.txt .

# Upgrade pip and install dependencies from `requirements.txt`
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . .

# Set environment variables for tokenizers (Avoid multiprocessing issues)
ENV TOKENIZERS_PARALLELISM=false

# Ensure PyTorch detects GPU
RUN python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Expose port (useful for API-based inference in the future)
EXPOSE 5000

# Default command (trains the model)
CMD ["python", "scripts/train.py"]
