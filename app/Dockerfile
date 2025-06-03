# 1. Base Image
FROM python:3.9-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# HuggingFace model name (can be overridden)
ARG HF_MODEL_NAME_ARG="distilbert-base-uncased-finetuned-sst-2-english"
ENV HF_MODEL_NAME=${HF_MODEL_NAME_ARG}

# Control model preloading
ENV PRELOAD_MODEL="true"

# Set working directory
WORKDIR /app

# 3. Install Python dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 4. Pre-download model into image
ENV TRANSFORMERS_CACHE=/app/models_cache
RUN mkdir -p ${TRANSFORMERS_CACHE}
RUN python - <<EOF
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = os.environ.get('HF_MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')
cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/app/models_cache')

print(f'Downloading model {model_name} to cache {cache_dir}...')
AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print('Model download complete.')
EOF

# 5. Copy application code
COPY . /app

# 6. Expose the port
EXPOSE 8000

# 7. Gunicorn Configs
ENV MODULE_NAME="app.main"
ENV VARIABLE_NAME="app"
ENV GUNICORN_CMD_ARGS="--bind 0.0.0.0:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker --log-level info --preload"

# 8. Run the app
CMD sh -c "gunicorn ${MODULE_NAME}:${VARIABLE_NAME} ${GUNICORN_CMD_ARGS}"
