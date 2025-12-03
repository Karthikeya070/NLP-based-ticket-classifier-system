# predict.py
import joblib
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer("sbert_encoder")

clf = joblib.load("ticket_classifier_model.joblib")

label_encoder = joblib.load("label_encoder.joblib")


def predict_ticket(subject: str, description: str) -> dict:
    text = (subject + " " + description).strip()

    if not text:
        return {"label": "Invalid input"}

    emb = sbert_model.encode([text])
    pred_encoded = clf.predict(emb)
    pred_label = label_encoder.inverse_transform(pred_encoded)

    return {"label": pred_label[0]}