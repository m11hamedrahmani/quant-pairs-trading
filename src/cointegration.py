"""
cointegration.py
----------------
Statistical tests and spread computation for pairs trading.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def test_cointegration(series1: pd.Series, series2: pd.Series, significance: float = 0.05) -> dict:
    """
    Run the Engle-Granger cointegration test on two price series.

    Parameters
    ----------
    series1, series2 : pd.Series
        Log price series for the two assets.
    significance : float
        p-value threshold (default 0.05).

    Returns
    -------
    dict with keys:
        - 'pvalue'      : float
        - 'cointegrated': bool
        - 'hedge_ratio' : float  (OLS beta)
        - 't_stat'      : float
    """
    log1 = np.log(series1)
    log2 = np.log(series2)

    # Engle-Granger test
    t_stat, pvalue, _ = coint(log1, log2)

    # OLS hedge ratio: log1 = alpha + beta * log2 + epsilon
    X = add_constant(log2.values)
    model = OLS(log1.values, X).fit()
    hedge_ratio = model.params[1]

    return {
        "pvalue": pvalue,
        "cointegrated": pvalue < significance,
        "hedge_ratio": hedge_ratio,
        "t_stat": t_stat,
    }


def compute_spread(series1: pd.Series, series2: pd.Series, hedge_ratio: float) -> pd.Series:
    """
    Compute the log-price spread: spread = log(S1) - hedge_ratio * log(S2)

    Parameters
    ----------
    series1, series2 : pd.Series
        Raw price series.
    hedge_ratio : float
        OLS beta from cointegration test.

    Returns
    -------
    pd.Series
        The spread time series.
    """
    log1 = np.log(series1)
    log2 = np.log(series2)
    spread = log1 - hedge_ratio * log2
    spread.name = "spread"
    return spread


def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute the rolling z-score of the spread.

    Parameters
    ----------
    spread : pd.Series
    window : int
        Rolling window in trading days (default 60).

    Returns
    -------
    pd.Series
        Z-score series (NaN for the first `window` observations).
    """
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    zscore = (spread - mean) / std
    zscore.name = "zscore"
    return zscore


def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity on a series.

    Returns
    -------
    dict with 'adf_stat', 'pvalue', 'stationary' (bool at 5% level)
    """
    result = adfuller(series.dropna())
    return {
        "adf_stat": result[0],
        "pvalue": result[1],
        "stationary": result[1] < 0.05,
    }
