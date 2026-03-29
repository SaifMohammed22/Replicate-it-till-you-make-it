"""
Fetch S&P 500 data from s-and-p-500-companies repo
"""

import click
import os
import pandas as pd
import random
import time
import urllib.request
import yfinance as yf
from datetime import datetime



DIR_PATH = os.path.dirname(os.path.relpath(__file__))
DATA_DIR = "data"
DATA_PATH = os.path.join(DIR_PATH, DATA_DIR)


SP500_LIST_URL = "https://github.com/datasets/s-and-p-500-companies/blob/main/data/constituents.csv"
SP500_LIST_PATH = os.path.join(DATA_PATH, "constituents.csv")

START_DATE = "1980-01-01"
now_date = datetime.now().strftime("%Y-%m-%d")


def _load_sp500():
    if os.path.exists(SP500_LIST_PATH):
        # Do nothing
        print("Data exists")
        return

    response = urllib.request.urlretrieve(SP500_LIST_URL, SP500_LIST_PATH)
    print("Downloading.....")
    if response:
        print("Downloaded successfully.") 

def _load_symbols():
    _load_sp500()
    df = pd.read_csv(SP500_LIST_PATH)
    symbols = df['Symbol'].unique().tolist()
    print(f"Loaded {len(symbols)} stock symbols")
    return symbols

def fetch_prices(symbol, output_path):
    """
    Fetch prices for stock 'symbol', from 1980-1-1 till today
    """
    stock = yf.Ticker(symbol)
    historical_data = stock.history(start=START_DATE, end=now_date)
    print(f"Fetching {symbol} stock prices .........")
    historical_data.to_csv(f"{output_path}/{symbol}.csv", index=False)

    

if __name__ == "__main__":
    print(DIR_PATH ,DATA_PATH, SP500_LIST_PATH)
    symbols = _load_symbols()
    for symbol in symbols:
        fetch_prices(symbol, DATA_PATH)
    print("Done fetching stock prices.")