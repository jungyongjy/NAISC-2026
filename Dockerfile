FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default entrypoint — callers must pass --train_data_filepath and --test_data_filepath
ENTRYPOINT ["python", "./src/main.py"]
