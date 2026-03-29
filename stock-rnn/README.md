# Stock Market Prices Prediction using RNN with Pytorch

This project is a modernized **Pytorch** version of [Lilian Weng](https://github.com/lilianweng/stock-rnn) stock market predication project from 2017. Following her two part tutorials [Part1](https://lilianweng.github.io/posts/2017-07-08-stock-rnn-part-1/) [Part2](https://lilianweng.github.io/lil-log/2017/07/22/predict-stock-prices-using-RNN-part-2.html)


## How to run it:

First, load the data from **fetch_data.py** by running:
```bash
python3 fetch_data.py
```
It will load 503 stock symbols and then fetch each symbol's stock price, starting from January 1, 1980, until the current date.

>Note: It will take a couple of minutes to fetch all the data. 
