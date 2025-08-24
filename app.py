# Install dependencies and libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the HF model
MODEL_NAME = "herrerovir/product-category-classifier-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the input data schema expected by the API
class TextIn(BaseModel):
    text: str

# Define the output data schema returned by the API
class PredictionOut(BaseModel):
    label: str
    confidence: float

# A simple root endpoint to verify the API is running
@app.get("/")
async def root():
    return {"message": "Product Category Classifier API is up!"}

# Main prediction endpoint
@app.post("/predict", response_model = PredictionOut)
async def predict_category(data: TextIn):
    if not data.text.strip():
        raise HTTPException(status_code = 400, detail = "Input text cannot be empty.")
    try:
        inputs = tokenizer(data.text, return_tensors = "pt", truncation = True, padding = True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim = -1)
        predicted_class_id = torch.argmax(probs).item()
        confidence = probs[0][predicted_class_id].item()
        labels = model.config.id2label
        predicted_label = labels[predicted_class_id]
        return PredictionOut(label = predicted_label, confidence = confidence)
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
