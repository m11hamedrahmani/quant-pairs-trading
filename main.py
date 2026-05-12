"""
main.py
-------
Entry point: run the full pairs trading pipeline end-to-end.

Usage
-----
    python main.py

Edit the CONFIG section below to change the pair, dates, or thresholds.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_pair
from src.cointegration import test_cointegration, compute_spread, compute_zscore, adf_test
from src.strategy import generate_signals, compute_returns
from src.backtest import compute_metrics, print_metrics, plot_results

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKER1 = "GS"          # Goldman Sachs
TICKER2 = "MS"          # Morgan Stanley
START   = "2018-01-01"
END     = "2024-01-01"

ZSCORE_WINDOW      = 60    # rolling window for z-score (trading days)
ENTRY_THRESHOLD    = 2.0
EXIT_THRESHOLD     = 0.5
STOP_THRESHOLD     = 3.5
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"\n[1/5] Loading data for {TICKER1} and {TICKER2} ({START} → {END})...")
    prices = load_pair(TICKER1, TICKER2, START, END)
    print(f"      {len(prices)} trading days loaded.\n")

    print("[2/5] Testing cointegration (Engle-Granger)...")
    coint_result = test_cointegration(prices[TICKER1], prices[TICKER2])
    print(f"      p-value      : {coint_result['pvalue']:.4f}")
    print(f"      Cointegrated : {coint_result['cointegrated']}")
    print(f"      Hedge ratio  : {coint_result['hedge_ratio']:.4f}\n")

    if not coint_result["cointegrated"]:
        print("  ⚠  Pair is NOT cointegrated at 5% level — strategy may not be valid.")
        print("     Continuing anyway for demonstration purposes.\n")

    print("[3/5] Computing spread and z-score...")
    spread = compute_spread(prices[TICKER1], prices[TICKER2], coint_result["hedge_ratio"])
    zscore = compute_zscore(spread, window=ZSCORE_WINDOW)

    adf = adf_test(spread)
    print(f"      ADF stat     : {adf['adf_stat']:.4f}")
    print(f"      ADF p-value  : {adf['pvalue']:.4f}")
    print(f"      Stationary   : {adf['stationary']}\n")

    print("[4/5] Generating signals and computing P&L...")
    signals = generate_signals(
        zscore,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        stop_threshold=STOP_THRESHOLD,
    )
    pnl = compute_returns(spread, signals)

    n_long  = (signals["signal"] == "entry_long").sum()
    n_short = (signals["signal"] == "entry_short").sum()
    n_stops = (signals["signal"] == "stop").sum()
    print(f"      Long entries : {n_long}")
    print(f"      Short entries: {n_short}")
    print(f"      Stops hit    : {n_stops}\n")

    print("[5/5] Performance metrics:")
    metrics = compute_metrics(pnl)
    print_metrics(metrics, TICKER1, TICKER2)

    print("Generating chart...")
    plot_results(
        prices=prices,
        spread=spread,
        zscore=zscore,
        signals=signals,
        pnl=pnl,
        ticker1=TICKER1,
        ticker2=TICKER2,
        save_path="results/strategy_chart.png",
    )


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
