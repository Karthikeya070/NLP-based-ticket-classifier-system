from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_ticket

app = FastAPI(title="NLP Ticket Classification API")

class Ticket(BaseModel):
    subject: str
    description: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Ticket Classifier API running"}

@app.post("/predict")
def predict(ticket: Ticket):
    result = predict_ticket(ticket.subject, ticket.description)
    return result
