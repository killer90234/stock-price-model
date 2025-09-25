import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os
import numpy as np

def fetch_stock(stock: str, period: str = "7d", interval="1d"):
    
    symbol = stock.upper() + '.NS'
    data = yf.download(symbol, period=period, interval=interval)
    return data

def train_model(stock: str, data):
    
    model_file = f"{stock}_model.pkl"
    x = np.arange(len(data)).reshape(-1,1)
    y = data["Close"].values
    
    model = LinearRegression()
    model.fit(x,y)
    
    
    # save model
    os.makedirs("models", exist_ok=True)
    with open(model_file,"wb") as f:
        pickle.dump(model, f)
        
    return model_file


if __name__ == "__main__":
    stock = input("Enter stock name: ")
    data = fetch_stock(stock)
    
    if data.empty:
        print("No data found for the given stock.")
    else:
        model_file = train_model(stock, data)
        print("Model trained and saved successfully:", model_file)
