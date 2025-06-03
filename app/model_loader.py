from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

# Global variable to hold the loaded pipeline
# This ensures the model is loaded only once
nlp_pipeline = None
MODEL_NAME = os.getenv("HF_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")

def load_model_pipeline():
    """Loads the HuggingFace sentiment analysis pipeline."""
    global nlp_pipeline
    if nlp_pipeline is None:
        print(f"Loading model: {MODEL_NAME}...")
        try:
            # Forcing CPU usage here for broader compatibility in demo environments.
            # For GPU, set device=0 (or specific GPU ID).
            device = -1 # -1 for CPU, 0 for first GPU, etc.
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            nlp_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=device 
            )
            print(f"Model '{MODEL_NAME}' loaded successfully on {'CPU' if device == -1 else 'GPU'}.")
        except Exception as e:
            print(f"Error loading model '{MODEL_NAME}': {e}")
            # Potentially raise the exception or handle it as per requirements
            raise
    return nlp_pipeline

def predict(text: str):
    """Performs inference on the input text using the loaded pipeline."""
    pipeline_instance = load_model_pipeline()
    if not text or not isinstance(text, str):
        return {"error": "Invalid input text."}
    try:
        # The pipeline returns a list of dictionaries, e.g., [{'label': 'POSITIVE', 'score': 0.999}]
        result = pipeline_instance(text)
        return result[0] 
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"Failed to process sentiment. Details: {str(e)}"}

# Pre-load the model when this module is imported by Gunicorn workers
if os.getenv("PRELOAD_MODEL", "true").lower() == "true":
    load_model_pipeline()