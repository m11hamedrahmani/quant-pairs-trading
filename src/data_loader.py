"""
data_loader.py
--------------
Downloads and preprocesses historical price data using yfinance.
"""

import yfinance as yf
import pandas as pd


def load_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        e.g. ['GS', 'MS']
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str
        End date in 'YYYY-MM-DD' format

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna()
    return prices


def load_pair(ticker1: str, ticker2: str, start: str, end: str) -> pd.DataFrame:
    """
    Convenience wrapper: load prices for exactly two tickers.
    """
    return load_prices([ticker1, ticker2], start=start, end=end)
