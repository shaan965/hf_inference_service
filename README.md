# HuggingFace Inference API (Containerized)

This project provides a scalable, containerized REST API for performing sentiment analysis using a pre-trained HuggingFace model. Built with FastAPI, Gunicorn, Uvicorn, and Docker, it is optimized to handle **multiple parallel inference requests** efficiently in a production-ready architecture.

---

## Features

- Uses `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace for sentiment classification.
- FastAPI + Uvicorn for async request handling.
- Gunicorn for multi-worker concurrency.
- Dockerized for consistent deployment.
- Includes a test notebook demonstrating parallel POST requests using `aiohttp` and `asyncio`.

---

## Model Info

We use the [DistilBERT SST-2 model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english), selected for:

- Fast load time and small size.
- Easy-to-interpret sentiment output (POSITIVE / NEGATIVE).
- Seamless integration with HuggingFace's `pipeline` API.

---

## Project Structure

### hf_inference_service/
│
├── app/
│ ├── main.py # FastAPI app with endpoints
│ ├── model_loader.py # Loads and manages HuggingFace pipeline
│ └── Dockerfile # Container configuration
│
├── notebook/
│ └── demo_parallel_reqs.ipynb # Jupyter Notebook for testing parallel requests
│
├── requirements.txt # Python dependencies
└── README.md
###
---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/shaan965/hf_inference_service.git
```

### 2. Build the Docker image
```
docker build -t hf-inference-app -f app/Dockerfile .
```

### 3. Run the container
```
docker run -d -p 8000:8000 --name sentiment-api hf-inference-app
```
### 4. Test the project 
```
curl http://localhost:8000/health
```

## Run the notebook 
```
cd notebook/
jupyter notebook demo_parallel_requests.ipynb
```

## System Architecture
- FastAPI serves the REST API.
- Uvicorn enables async request processing.
- Gunicorn spawns multiple worker processes.
- Docker containers wrap everything for consistency and portability.
- aiohttp and asyncio simulate parallel client requests.
