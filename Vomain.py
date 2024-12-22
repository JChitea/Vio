import numpy as np
import pandas as pd
import yfinance as yf

class watchList(object):
    """Create a watchlist object to store poitners to different tickers"""
    def __init__(self, ticklst = []):
        if len(ticklst) != 0:
            ticks = [get_stock(ticker) for ticker in ticklst]
            self.W = dict(zip(ticklst, ticks))
        else:
            self.W = {}

def get_stock(ticker):
    """Create a yfinance ticker object"""
    return yf.Ticker(ticker)

# We'll need a watchlist, a PL tracker (interesting to think about how this will be structured), An AI agent entry point, config control