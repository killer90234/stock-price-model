from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict import make_predication

app = FastAPI()

@app.get("/")
def home():
    return {"Welcome to the Stock Price Prediction API"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/predict/")
def predict(stock: str):
    prediction = make_predication(stock)
    return prediction
   
