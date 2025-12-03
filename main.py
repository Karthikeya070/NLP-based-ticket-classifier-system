# predict.py

import joblib
from sentence_transformers import SentenceTransformer

# Load OFFLINE SBERT model ONCE
print("Loading embeddings model...")
sbert_model = SentenceTransformer("models/all-MiniLM-L6-v2")
print("SBERT loaded.\n")

# Load classifier
print("Loading classifier...")
clf = joblib.load("ticket_classifier_model.joblib")
print("Classifier loaded.\n")

# Load label encoder
print("Loading label encoder...")
label_encoder = joblib.load("label_encoder.joblib")
print("Label encoder loaded.\n")


def predict_ticket(subject: str, description: str) -> dict:
    """Predict the ticket type based on subject + description."""
    
    # Combine fields
    text = subject + " " + description
    
    # Encode using SBERT
    emb = sbert_model.encode([text])
    
    # Predict
    pred = clf.predict(emb)[0]
    
    # Convert back to label
    label = label_encoder.inverse_transform([pred])[0]
    
    return {"prediction": label}
