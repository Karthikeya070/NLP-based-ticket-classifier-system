import pandas as pd
import joblib
import json
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# CONFIG
# -----------------------------
LOCAL_MODEL_PATH = "./sbert_encoder"
CLASSIFIER_FILE = "ticket_classifier_model.joblib"
LABEL_FILE = "label_encoder.joblib"
DATA_FILE = "customer_support_tickets.csv"
METRICS_FILE = "metrics.json"

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="NLP Ticket Classifier API")

@app.get("/")
def home():
    return {"message": "Ticket Classifier API Running!"}

# -----------------------------
# REQUEST MODEL
# -----------------------------
class TicketInput(BaseModel):
    subject: str
    description: str

# -----------------------------
# LOAD MODELS AT STARTUP
# -----------------------------
sbert_model = SentenceTransformer(LOCAL_MODEL_PATH)
clf = joblib.load(CLASSIFIER_FILE)
label_encoder = joblib.load(LABEL_FILE)

# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
def predict_ticket(data: TicketInput):

    # SUBJECT IS KEPT FOR UX, BUT NOT USED FOR TRAINING
    text = data.description.strip()

    if not text:
        return {"error": "Empty description"}

    emb = sbert_model.encode([text])
    pred = clf.predict(emb)[0]
    label = label_encoder.inverse_transform([pred])[0]

    return {"prediction": label}

# -----------------------------------------------------
# TRAINING FUNCTION (RUN LOCALLY)
# -----------------------------------------------------
def train_new_model():

    df = pd.read_csv(DATA_FILE)

    # --------- CRITICAL FIXES ---------
    # 1. Use description only (removes label leakage)
    df["text"] = df["Ticket Description"]

    # 2. Drop duplicates
    df = df.drop_duplicates(subset=["text", "Ticket Type"])

    X = df["text"].tolist()
    y = df["Ticket Type"].tolist()

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Embeddings AFTER split (important)
    sbert = SentenceTransformer(LOCAL_MODEL_PATH)
    X_train_emb = sbert.encode(X_train, show_progress_bar=True)
    X_test_emb = sbert.encode(X_test, show_progress_bar=True)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_emb, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_emb)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Save artifacts
    joblib.dump(clf, CLASSIFIER_FILE)
    joblib.dump(label_encoder, LABEL_FILE)

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nTraining complete (LEAKAGE FIXED)")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

# -----------------------------------------------------
# LOCAL TRAINING
# -----------------------------------------------------
if __name__ == "__main__":
    train_new_model()
