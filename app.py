import streamlit as st
import requests

API_URL = "https://nlp-based-ticket-classifier-system.onrender.com/predict"

st.set_page_config(page_title="Ticket Classifier", layout="centered")

st.title("🎫 NLP Based Customer Support Ticket Classifier")
st.write("Enter the ticket subject and description to classify the ticket.")

subject = st.text_input("Subject")
description = st.text_area("Description")

if st.button("Classify"):
    if not subject or not description:
        st.error("Please fill both subject and description!")
    else:
        with st.spinner("Classifying..."):
            payload = {"subject": subject, "description": description}

            try:
                response = requests.post(API_URL, json=payload)
                result = response.json()
            except Exception as e:
                st.error("❌ Cannot reach API server.")
                st.write(e)
                st.stop()

            # --- Show result as Category ---
            if "prediction" in result:
                st.success(f"Category: **{result['prediction']}**")
            else:
                st.error("❌ API Error occurred!")
                st.write(result)  # For debugging
