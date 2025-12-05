from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
import os # Import os for setting the correct path

# ==============================
# Define Local Model Path
# ==============================
# IMPORTANT: Point this to the local folder containing pytorch_model.bin, config.json, etc.
# Based on your file structure, this is likely 'sbert_encoder'. Adjust if needed.
LOCAL_MODEL_PATH = "./sbert_encoder"

# ==============================
# Load Models ONCE on startup
# ==============================
print("Loading embeddings model...")

# The SentenceTransformer constructor automatically detects local files.
# We ensure it looks at the local folder we already pushed.
try:
    sbert_model = SentenceTransformer(LOCAL_MODEL_PATH)
    print("SBERT loaded from local files successfully.")
except Exception as e:
    print(f"Error loading local model: {e}")
    print("Please ensure the directory './sbert_encoder' exists and contains all model files.")
    raise

print("Loading classifier...")
# Ensure these joblib files are also present in the deployment environment
clf = joblib.load("ticket_classifier_model.joblib")
print("Classifier loaded.\n")

print("Loading label encoder...")
label_encoder = joblib.load("label_encoder.joblib")
print("Label encoder loaded.\n")


# ==============================
# Prediction Function
# ==============================
def predict_ticket(subject: str, description: str) -> dict:
    """Predict the ticket type based on subject + description."""
    
    text = subject + " " + description
    # SentenceTransformer.encode expects a list of strings
    emb = sbert_model.encode([text])
    pred = clf.predict(emb)[0]
    label = label_encoder.inverse_transform([pred])[0]
    
    return {"prediction": label}


# ==============================
# FASTAPI APP
# ==============================
app = FastAPI()

class Ticket(BaseModel):
    subject: str
    description: str


@app.get("/")
def home():
    return {"status": "API running successfully"}


@app.post("/predict")
def predict(ticket: Ticket):
    return predict_ticket(ticket.subject, ticket.description)