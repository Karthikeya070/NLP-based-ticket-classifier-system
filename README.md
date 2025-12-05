NLP Based Ticket Classifier

A machine learning system that automatically classifies customer support tickets based on the subject and description. It uses a lightweight MiniLM Sentence Transformer for semantic understanding and a Logistic Regression model for prediction. The backend runs on FastAPI and is deployed on Render as a public API.

How It Works

Subject + description are combined into a single text.

MiniLM sentence transformer converts text into embeddings.

Logistic Regression predicts the ticket category.

FastAPI returns the result as JSON.

Features

Fast and lightweight transformer-based classification

Clean FastAPI backend with JSON responses

Easily deployable on Render

Optional Streamlit interface for end users

Project Structure
main.py          → FastAPI app  
predict.py       → Prediction logic  
sbert_encoder/   → Saved MiniLM model  
app.py           → Streamlit UI (optional)  
requirements.txt → Dependencies  
render.yaml      → Deployment config  

API Endpoints
GET /

Health check endpoint.

POST /predict

Input:

{
  "subject": "Unable to log in",
  "description": "The login page keeps refreshing"
}


Output:

{
  "category": "Login Issue"
}

Running Locally
pip install -r requirements.txt
uvicorn main:app --reload


API URL:

http://127.0.0.1:8000

Deployment (Render)

FastAPI runs as a web service on Render

Render installs dependencies, loads the MiniLM model, and keeps the API available

Streamlit Frontend (Optional)

A simple UI for users to test the classifier.

app.py example:

import streamlit as st
import requests

API_URL = "YOUR_RENDER_API_URL/predict"

st.title("Ticket Classifier")

subject = st.text_input("Subject")
description = st.text_area("Description")

if st.button("Predict"):
    response = requests.post(API_URL, json={"subject": subject, "description": description})
    st.write("Prediction:", response.json().get("prediction"))


Run:

streamlit run app.py

Future Enhancements

Expand categories

Improve training dataset

Enhanced UI

Analytics for ticket trends