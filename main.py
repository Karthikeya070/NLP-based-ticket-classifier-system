# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer

# ==============================
# Load Models ONCE on startup
# ==============================
print("Loading embeddings model...")
sbert_model = SentenceTransformer("models/all-MiniLM-L6-v2")

print("SBERT loaded.\n")

print("Loading classifier...")
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
