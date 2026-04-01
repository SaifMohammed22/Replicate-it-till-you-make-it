"""
Make stock dataset out of the fetched data
"""
import os
import numpy as np
import pandas as pd
import random


class StockDataSet:
    def __init__(self, 
                 stock_symbol: str,
                 input_size=1,
                 num_steps=30,
                 test_ratio=0.2,
                 normalized=True,
                 close_price_only=True
                 ):
        self.stock_symbol = stock_symbol
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.normalized = normalized
        self.close_price_only = close_price_only


        # Read data
        raw_df = pd.read_csv(f"stock-rnn/data/{self.stock_symbol}.csv")


        if self.close_price_only:
            self.raw_seq = raw_df['Close'].tolist()
        else:
            self.raw_seq = [price for tup in raw_df[['Open', 'Close']].values for price in tup]

        self.raw_seq = np.array(self.raw_seq)
        self.train_X, self.test_X, self.train_y, self.test_y = self._prep_data(self.raw_seq)

    def info(self):
        print(f"StockDataset {self.stock_symbol} train: {len(self.train_X)} test: {len(self.test_X)}")
        
    def _prep_data(self, seq):
        """Split data into train and test with a specific test_ratio"""
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]
        
        X = [seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)]
        y = [(seq[i + self.num_steps] for i in range(len(seq) - self.num_steps))]

        train_size = int((1 - self.test_ratio) * len(X))
        train_X, test_X = X[: train_size], X[train_size:]
        train_y, test_y = y[: train_size], y[train_size:]

        return train_X, test_X, train_y, test_y 

if __name__ == "__main__":
    stock = StockDataSet("A", close_price_only=True)
    print(len(stock.raw_seq))
    print(stock.raw_seq[:10])
    print(len(stock.train_X), len(stock.test_X))


        
