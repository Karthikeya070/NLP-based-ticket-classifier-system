# NLP Based Ticket Classifier

This project automatically classifies customer support tickets based on the subject and description provided by the user. The purpose of the system is to help support teams sort tickets into categories such as refund requests, technical issues, payment problems, account related issues, and similar topics. This improves response time and keeps the support workflow organised.

The system uses a sentence transformer model to understand the meaning of the text. It does not depend on plain keywords. Instead, it looks at the context and intent behind the message. This gives much better accuracy, especially when users describe the same issue in different ways.

The backend is built using FastAPI. It is deployed on Render as a web service. The API receives ticket text and returns the predicted category.

---

## How the model works

1. The subject and description of the ticket are combined into a single input string.
2. A sentence transformer model (MiniLM) converts this text into a numerical vector. This vector represents the meaning of the sentence.
3. A logistic regression model uses these vectors to decide the correct category.
4. The API sends back the predicted category in JSON format.

The MiniLM model is lightweight and fast, which makes it suitable for deployment on Render while still giving strong accuracy.

---

## Features

- Classifies tickets into predefined categories  
- Uses a compact and efficient MiniLM sentence transformer  
- Fast prediction using FastAPI  
- Clean and simple codebase  
- Can be deployed easily on Render  

---

## Project Structure

main.py FastAPI application
predict.py Prediction function using the trained model
sbert_model.py SBERT related logic
sbert_encoder/ Saved MiniLM model
requirements.txt Python dependencies
render.yaml Render deployment configuration
eda.py Exploratory data analysis (optional)
test_api.py Script for testing API locally


---

## API Endpoints

### GET `/`
Returns a small message to show that the API is running.

### POST `/predict`
Accepts subject and description as input, then returns the predicted ticket category.

#### Example request

```json
{
  "subject": "Unable to log in",
  "description": "The login page keeps refreshing"
}

Example response
{
  "category": "Login Issue"
}

How to run locally

Create and activate a virtual environment

Install dependencies:

pip install -r requirements.txt


Start the FastAPI server:

uvicorn main:app --reload


Open the API in a browser:

http://127.0.0.1:8000

Deployment

The project is deployed on Render using the render.yaml file. Render installs dependencies, loads the MiniLM model from the repository, starts the server, and keeps the service active.

Future improvements

1)Adding more ticket categories

2)Improving and expanding training data

3)Adding a simple frontend for end users

4)Adding analytics to track common ticket trends