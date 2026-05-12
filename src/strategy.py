"""
strategy.py
-----------
Signal generation for the mean-reversion pairs trading strategy.

Entry / Exit logic
------------------
  - LONG  spread (buy S1, sell S2) when z-score < -entry_threshold
  - SHORT spread (sell S1, buy S2) when z-score >  entry_threshold
  - EXIT position                  when |z-score| < exit_threshold
  - STOP  position                 when |z-score| > stop_threshold  (risk control)
"""

import pandas as pd
import numpy as np


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Generate long/short/flat signals from the z-score.

    Parameters
    ----------
    zscore           : pd.Series
    entry_threshold  : float  — enter trade when |z| crosses this level
    exit_threshold   : float  — close trade when |z| falls below this level
    stop_threshold   : float  — hard stop-loss when |z| exceeds this level

    Returns
    -------
    pd.DataFrame with columns:
        - 'zscore'   : original z-score
        - 'position' : +1 (long spread), -1 (short spread), 0 (flat)
        - 'signal'   : trade trigger at each bar ('entry_long', 'entry_short', 'exit', 'stop', '')
    """
    df = pd.DataFrame({"zscore": zscore})
    df["position"] = 0
    df["signal"] = ""

    position = 0

    for i in range(1, len(df)):
        z = df["zscore"].iloc[i]

        if np.isnan(z):
            df.iloc[i, df.columns.get_loc("position")] = 0
            continue

        # --- Stop loss ---
        if position != 0 and abs(z) > stop_threshold:
            position = 0
            df.iloc[i, df.columns.get_loc("signal")] = "stop"

        # --- Exit ---
        elif position != 0 and abs(z) < exit_threshold:
            position = 0
            df.iloc[i, df.columns.get_loc("signal")] = "exit"

        # --- Entry ---
        elif position == 0:
            if z < -entry_threshold:
                position = 1          # long spread
                df.iloc[i, df.columns.get_loc("signal")] = "entry_long"
            elif z > entry_threshold:
                position = -1         # short spread
                df.iloc[i, df.columns.get_loc("signal")] = "entry_short"

        df.iloc[i, df.columns.get_loc("position")] = position

    return df


def compute_returns(
    spread: pd.Series,
    signals: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily strategy P&L from spread changes and position.

    The spread return on day t is:
        r_t = spread_t - spread_{t-1}
    The strategy return is:
        strat_t = position_{t-1} * r_t

    Returns
    -------
    pd.Series of daily P&L (in spread units).
    """
    spread_returns = spread.diff()
    pnl = signals["position"].shift(1) * spread_returns
    pnl.name = "pnl"
    return pnl.dropna()
