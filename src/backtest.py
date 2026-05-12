"""
backtest.py
-----------
Performance analytics and visualisation for the pairs trading strategy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ── Performance metrics ──────────────────────────────────────────────────────

def sharpe_ratio(pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    if pnl.std() == 0:
        return 0.0
    return (pnl.mean() / pnl.std()) * np.sqrt(periods_per_year)


def max_drawdown(pnl: pd.Series) -> float:
    """Maximum drawdown of the cumulative P&L curve."""
    cum = pnl.cumsum()
    rolling_max = cum.cummax()
    drawdown = cum - rolling_max
    return drawdown.min()


def win_rate(pnl: pd.Series) -> float:
    """Fraction of days with positive P&L."""
    return (pnl > 0).sum() / len(pnl)


def compute_metrics(pnl: pd.Series) -> dict:
    """
    Aggregate all performance metrics into a single dictionary.

    Returns
    -------
    dict with keys:
        total_pnl, annualised_sharpe, max_drawdown,
        win_rate, num_days, avg_daily_pnl
    """
    return {
        "total_pnl": pnl.sum(),
        "annualised_sharpe": sharpe_ratio(pnl),
        "max_drawdown": max_drawdown(pnl),
        "win_rate": win_rate(pnl),
        "num_days": len(pnl),
        "avg_daily_pnl": pnl.mean(),
    }


def print_metrics(metrics: dict, ticker1: str = "S1", ticker2: str = "S2") -> None:
    """Pretty-print strategy performance metrics."""
    print(f"\n{'='*50}")
    print(f"  Pairs Trading Strategy: {ticker1} / {ticker2}")
    print(f"{'='*50}")
    print(f"  Total P&L            : {metrics['total_pnl']:.4f}")
    print(f"  Annualised Sharpe    : {metrics['annualised_sharpe']:.3f}")
    print(f"  Max Drawdown         : {metrics['max_drawdown']:.4f}")
    print(f"  Win Rate             : {metrics['win_rate']:.1%}")
    print(f"  Trading Days         : {metrics['num_days']}")
    print(f"  Avg Daily P&L        : {metrics['avg_daily_pnl']:.6f}")
    print(f"{'='*50}\n")


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(
    prices: pd.DataFrame,
    spread: pd.Series,
    zscore: pd.Series,
    signals: pd.DataFrame,
    pnl: pd.Series,
    ticker1: str,
    ticker2: str,
    save_path: str = None,
) -> None:
    """
    Four-panel chart:
      1. Normalised prices of the two assets
      2. Spread with rolling mean ± 1 std band
      3. Z-score with entry/exit thresholds and trade signals
      4. Cumulative P&L
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    fig.suptitle(
        f"Pairs Trading Strategy: {ticker1} / {ticker2}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # ── Panel 1: Normalised prices ──
    ax1 = axes[0]
    norm = prices / prices.iloc[0] * 100
    ax1.plot(norm.index, norm.iloc[:, 0], label=ticker1, color="#1f77b4", linewidth=1.2)
    ax1.plot(norm.index, norm.iloc[:, 1], label=ticker2, color="#ff7f0e", linewidth=1.2)
    ax1.set_ylabel("Normalised Price (base 100)")
    ax1.set_title("Asset Prices")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Spread ──
    ax2 = axes[1]
    roll_mean = spread.rolling(60).mean()
    roll_std = spread.rolling(60).std()
    ax2.plot(spread.index, spread, label="Spread", color="#2ca02c", linewidth=1.0)
    ax2.plot(roll_mean.index, roll_mean, color="black", linewidth=1.0, linestyle="--", label="Rolling Mean")
    ax2.fill_between(
        spread.index,
        (roll_mean - roll_std),
        (roll_mean + roll_std),
        alpha=0.15,
        color="gray",
        label="±1 Std",
    )
    ax2.set_ylabel("Log Spread")
    ax2.set_title("Spread with Rolling Mean ± 1 Std")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Z-score + signals ──
    ax3 = axes[2]
    ax3.plot(zscore.index, zscore, label="Z-score", color="#9467bd", linewidth=1.0)
    ax3.axhline(2.0,  color="red",   linestyle="--", linewidth=0.9, label="Entry ±2.0")
    ax3.axhline(-2.0, color="red",   linestyle="--", linewidth=0.9)
    ax3.axhline(0.5,  color="green", linestyle=":",  linewidth=0.9, label="Exit ±0.5")
    ax3.axhline(-0.5, color="green", linestyle=":",  linewidth=0.9)
    ax3.axhline(3.5,  color="black", linestyle="-.", linewidth=0.9, label="Stop ±3.5")
    ax3.axhline(-3.5, color="black", linestyle="-.", linewidth=0.9)
    ax3.axhline(0,    color="gray",  linestyle="-",  linewidth=0.5)

    # Plot entry/exit markers
    entries_long  = signals[signals["signal"] == "entry_long"].index
    entries_short = signals[signals["signal"] == "entry_short"].index
    exits         = signals[signals["signal"].isin(["exit", "stop"])].index

    ax3.scatter(entries_long,  zscore.reindex(entries_long),  marker="^", color="blue",  s=40, zorder=5, label="Long Entry")
    ax3.scatter(entries_short, zscore.reindex(entries_short), marker="v", color="red",   s=40, zorder=5, label="Short Entry")
    ax3.scatter(exits,         zscore.reindex(exits),         marker="x", color="black", s=40, zorder=5, label="Exit/Stop")

    ax3.set_ylabel("Z-score")
    ax3.set_title("Z-score and Trade Signals")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Cumulative P&L ──
    ax4 = axes[3]
    cum_pnl = pnl.cumsum()
    ax4.plot(cum_pnl.index, cum_pnl, label="Cumulative P&L", color="#d62728", linewidth=1.5)
    ax4.axhline(0, color="black", linewidth=0.6)
    ax4.fill_between(cum_pnl.index, cum_pnl, 0, where=(cum_pnl >= 0), alpha=0.2, color="green")
    ax4.fill_between(cum_pnl.index, cum_pnl, 0, where=(cum_pnl < 0),  alpha=0.2, color="red")
    ax4.set_ylabel("Cumulative P&L (log spread units)")
    ax4.set_title("Strategy Cumulative P&L")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()
