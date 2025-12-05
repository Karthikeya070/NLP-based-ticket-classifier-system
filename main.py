import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- Configuration ---
LOCAL_MODEL_PATH = "./sbert_encoder"   # Your local 384-dim SBERT model folder
DATA_FILE = "customer_support_tickets.csv"  
CLASSIFIER_OUTPUT = "ticket_classifier_model.joblib"
LABEL_ENCODER_OUTPUT = "label_encoder.joblib"


def load_data():
    """Load data from CSV and combine subject + description."""
    print(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"❌ ERROR: {DATA_FILE} not found. Check your file path.")
        return None, None

    print("\nCSV Columns Found:", list(df.columns), "\n")

    # --- FIXED COLUMN NAMES BASED ON YOUR CSV ---
    if not {"Ticket Subject", "Ticket Description", "Ticket Type"}.issubset(df.columns):
        print("❌ ERROR: Your CSV must contain these columns:")
        print("   • Ticket Subject")
        print("   • Ticket Description")
        print("   • Ticket Type")
        return None, None

    # Combine text fields
    df["text"] = df["Ticket Subject"] + " " + df["Ticket Description"]
    
    # Inputs (text) and labels (category)
    X_text = df["text"].tolist()
    y_labels = df["Ticket Type"].tolist()

    return X_text, y_labels


def train_new_model():
    """Train Logistic Regression using 384-dim SBERT embeddings."""
    
    # 1. Load the dataset
    X_text, y_labels = load_data()
    if X_text is None:
        return
    
    # 2. Load local 384-dim SBERT model
    print(f"Loading SBERT model from {LOCAL_MODEL_PATH}...")
    try:
        sbert_model = SentenceTransformer(LOCAL_MODEL_PATH)
    except Exception as e:
        print("\n❌ FATAL ERROR: Could not load the SBERT model.")
        print("Make sure the folder './sbert_encoder' exists and contains:")
        print("   - config.json")
        print("   - sentence_bert_config.json")
        print("   - pytorch_model.bin (or safetensors)")
        print("   - tokenizer files\n")
        print(e)
        return

    # 3. Convert all texts into embeddings
    print("\nGenerating 384-dimensional embeddings...")
    X_embeddings = sbert_model.encode(X_text, show_progress_bar=True)
    print(f"Embedding shape: {X_embeddings.shape} (should be N x 384)\n")

    # 4. Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)

    # 5. Train-test split (for validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y_encoded, 
        test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 6. Train logistic regression classifier
    print("\nTraining Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    # 7. Check validation accuracy
    accuracy = clf.score(X_test, y_test)
    print(f"\n✅ Validation Accuracy: {accuracy:.4f}\n")

    # 8. Retrain on full dataset before saving
    print("Retraining classifier on full dataset...")
    clf.fit(X_embeddings, y_encoded)

    # 9. Save classifier + label encoder
    joblib.dump(clf, CLASSIFIER_OUTPUT)
    joblib.dump(label_encoder, LABEL_ENCODER_OUTPUT)

    print("\n🎉 SUCCESS! Model training complete.")
    print(f"Saved classifier → {CLASSIFIER_OUTPUT}")
    print(f"Saved label encoder → {LABEL_ENCODER_OUTPUT}")
    print("\nYour new model now expects **384-dimension embeddings**, matching your local SBERT model.")


if __name__ == "__main__":
    train_new_model()
