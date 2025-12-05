import streamlit as st
import requests

API_URL = "https://nlp-based-ticket-classifier-system.onrender.com/predict"

st.set_page_config(page_title="Ticket Classifier", layout="centered")

st.title("🎫 NLP Ticket Classifier")
st.write("Enter the ticket subject and description to classify the ticket.")

subject = st.text_input("Subject")
description = st.text_area("Description")

if st.button("Predict"):
    if not subject or not description:
        st.error("Please fill both subject and description!")
    else:
        with st.spinner("Predicting..."):
            payload = {"subject": subject, "description": description}
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['prediction']}**")
            else:
                st.error("API Error! Check logs.")
