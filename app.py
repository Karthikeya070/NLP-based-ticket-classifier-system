import streamlit as st
import requests
import json
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "https://nlp-based-ticket-classifier-system.onrender.com/predict"
METRICS_FILE = "metrics.json"

st.set_page_config(
    page_title="Ticket Classifier",
    layout="centered"
)

st.title("🎫 NLP Based Customer Support Ticket Classifier")
st.write("Enter the ticket subject and description to classify the ticket.")

# -----------------------------
# INPUTS
# -----------------------------
subject = st.text_input("Subject")
description = st.text_area("Description")

if st.button("Classify"):
    if not subject or not description:
        st.error("Please fill both subject and description!")
    else:
        with st.spinner("Classifying..."):
            payload = {
                "subject": subject,
                "description": description
            }

            try:
                response = requests.post(API_URL, json=payload)
                result = response.json()
            except Exception as e:
                st.error("❌ Cannot reach API server.")
                st.write(e)
                st.stop()

            if "prediction" in result:
                st.success(f"Category: **{result['prediction']}**")
            else:
                st.error("❌ API Error")
                st.write(result)

# -----------------------------
# METRICS SECTION
# -----------------------------
st.divider()
st.subheader("📊 Model Evaluation Metrics")

try:
    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    # ---- Accuracy (shown separately, correctly) ----
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")

    # ---- Classification Report ----
    report_df = pd.DataFrame(metrics["classification_report"]).T

    # Remove accuracy row (breaks table structure)
    report_df = report_df.drop(index="accuracy")

    # Convert index to column for proper UI rendering
    report_df = report_df.reset_index()
    report_df = report_df.rename(columns={"index": "Class"})

    # Round numeric values
    report_df[["precision", "recall", "f1-score"]] = report_df[
        ["precision", "recall", "f1-score"]
    ].round(3)

    # Reorder columns
    report_df = report_df[
        ["Class", "precision", "recall", "f1-score", "support"]
    ]

    st.subheader("Classification Report")
    st.dataframe(report_df, width="stretch")

    # ---- Confusion Matrix ----
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(metrics["confusion_matrix"])
    st.dataframe(cm_df, width="stretch")

except FileNotFoundError:
    st.info("Metrics not available. Run training locally to generate metrics.")
except Exception as e:
    st.error("Error loading metrics.")
    st.write(e)
