import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# -----------------------------
# CONFIG
# -----------------------------
LOCAL_MODEL_PATH = "./sbert_encoder"
CLASSIFIER_FILE = "ticket_classifier_model.joblib"
LABEL_FILE = "label_encoder.joblib"
DATA_FILE = "customer_support_tickets.csv"


# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI()


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
# LOAD MODELS ONCE AT STARTUP
# -----------------------------
print("Loading SBERT model...")
sbert_model = SentenceTransformer(LOCAL_MODEL_PATH)

print("Loading classifier...")
clf = joblib.load(CLASSIFIER_FILE)

print("Loading label encoder...")
label_encoder = joblib.load(LABEL_FILE)

print("All models loaded successfully!\n")


# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
def predict_ticket(data: TicketInput):

    # Combine text
    text = data.subject + " " + data.description

    # Encode into 384D embedding
    try:
        emb = sbert_model.encode([text])
    except Exception as e:
        return {"error": "Embedding generation failed", "details": str(e)}

    # Predict
    try:
        pred = clf.predict(emb)[0]
        label = label_encoder.inverse_transform([pred])[0]
    except Exception as e:
        return {"error": "Prediction failed", "details": str(e)}

    return {"prediction": label}


# -----------------------------------------------------
# TRAINING FUNCTION (You run manually, NOT in Render)
# -----------------------------------------------------
def train_new_model():

    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    df["text"] = df["Ticket Subject"] + " " + df["Ticket Description"]
    X = df["text"].tolist()
    y = df["Ticket Type"].tolist()

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Loading SBERT for embeddings...")
    sbert = SentenceTransformer(LOCAL_MODEL_PATH)
    X_emb = sbert.encode(X, show_progress_bar=True)

    print("Training classifier...")
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_emb, y_encoded)

    print("Saving classifier + label encoder...")
    joblib.dump(clf, CLASSIFIER_FILE)
    joblib.dump(label_encoder, LABEL_FILE)

    print("Training complete!")


if __name__ == "__main__":
    train_new_model()
