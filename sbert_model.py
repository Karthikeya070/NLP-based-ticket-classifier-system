# sbert_model.py
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("customer_support_tickets.csv")

# Load label encoder
label_encoder = joblib.load("label_encoder.joblib")

# Combine subject + description
X = df["Ticket Subject"] + " " + df["Ticket Description"]
y = label_encoder.transform(df["Ticket Type"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

# Load SBERT model
sbert_model = SentenceTransformer("all-mpnet-base-v2")
print("SBERT loaded successfully.\n")

# Convert text to embeddings
X_train_emb = sbert_model.encode(X_train.tolist(), batch_size=32, show_progress_bar=True)
X_test_emb  = sbert_model.encode(X_test.tolist(), batch_size=32, show_progress_bar=True)

# Classifier
clf = LogisticRegression(
    max_iter=5000,
    n_jobs=-1,
    verbose=1
)
clf.fit(X_train_emb, y_train)

# Evaluate model
y_pred = clf.predict(X_test_emb)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save classifier only
joblib.dump(clf, "ticket_classifier_model.joblib")

print("Classifier saved as ticket_classifier_model.joblib")
print("SBERT will NOT be saved locally — loaded from HF")

# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("customer_support_tickets.csv")

# NEW: Combine subject + description
df["text"] = df["Ticket Subject"].astype(str) + " " + df["Ticket Description"].astype(str)

X = df["text"]                    # changed line
y = df["Ticket Type"]

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Encoded labels:", y_encoded[:10])

# Show mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping, "\n")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Train size:", len(X_train))
print("Test size:", len(X_test), "\n")

# Save label encoder
joblib.dump(label_encoder, "label_encoder.joblib")
print("Label encoder saved.")

print("Unique encoded labels:", set(y_encoded))