from train import train_model, fetch_stock
import os
import numpy as np
import pickle


def make_predication(stock: str):
    stock = stock.upper()
    model_file = f"{stock}_model.pkl"
    
    data = fetch_stock(stock)
    if data.empty:
        return {"error": "No data found for the given stock."}
    
    if not os.path.exists(model_file):
        os.makedirs("models", exist_ok=True) 
        model_file = train_model(stock, data)

    with open(model_file, "rb") as f:
        model = pickle.load(f)
    
    last_day_index = len(data) - 1
    prediction = model.predict(np.array([[last_day_index + 1]]))[0]
    
    last_close = data["Close"].iloc[-1]
    
    return {
        "stock": stock,
        "last_close": round(last_close, 2),
        "predicted_next_close": round(float(prediction), 2)
    }
    
