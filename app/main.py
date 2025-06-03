from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os

# Import model loading and prediction functions from model_loader
from .model_loader import predict, MODEL_NAME, load_model_pipeline

# Pydantic models for request and response validation
class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=1, example="This is a wonderful demonstration!")

class InferenceResponse(BaseModel):
    label: str
    score: float

# Initialize FastAPI app
app = FastAPI(
    title="HuggingFace Model Inference API",
    description=f"API for inference using the '{MODEL_NAME}' model from HuggingFace.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """
    Event handler for application startup.
    Ensures the model is loaded when the first Uvicorn worker starts.
    Gunicorn might preload, but this is an additional check.
    """
    print("FastAPI application starting up...")
    if os.getenv("PRELOAD_MODEL", "true").lower() == "true":
        # Model is preloaded by model_loader.py if PRELOAD_MODEL is true
        print(f"Model '{MODEL_NAME}' should be preloaded by worker initialization.")
    else:
        # Explicitly load if not preloaded (though preloading is preferred for workers)
        print("Attempting to load model on startup event (if not already preloaded)...")
        load_model_pipeline() 
    print("FastAPI application startup complete.")


@app.post("/predict", response_model=InferenceResponse, tags=["Inference"])
async def get_prediction(request: InferenceRequest):
    """
    Accepts a single text input and returns its sentiment or other model output.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty or just whitespace.")
    
    prediction_result = predict(request.text)

    if "error" in prediction_result:
        raise HTTPException(status_code=500, detail=prediction_result["error"])
    
    return InferenceResponse(label=prediction_result['label'], score=prediction_result['score'])

@app.post("/predict_batch", response_model=List[InferenceResponse], tags=["Inference"])
async def get_batch_predictions(requests: List[InferenceRequest]):
    """
    Accepts a batch of text inputs and returns their sentiments.
    Note: This endpoint processes items sequentially internally within a single request.
    True parallelism for batch is achieved when multiple such requests hit different Gunicorn workers.
    """
    if not requests:
        raise HTTPException(status_code=400, detail="Input batch cannot be empty.")
    
    results = []
    for i, req_item in enumerate(requests):
        if not req_item.text.strip():
            # Handle invalid item in batch, e.g., skip or return error placeholder
            results.append(InferenceResponse(label="INVALID_INPUT", score=0.0))
            continue
        
        prediction_result = predict(req_item.text)
        if "error" in prediction_result:
            # Handle error for specific item in batch
            results.append(InferenceResponse(label="ERROR_PROCESSING", score=0.0))
        else:
            results.append(InferenceResponse(label=prediction_result['label'], score=prediction_result['score']))
            
    return results

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Basic health check endpoint."""
    # Could add a check to see if the model pipeline is loaded
    from .model_loader import nlp_pipeline # Check current state
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_loaded": "yes" if nlp_pipeline is not None else "no"
    }