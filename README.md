# Quantitative Pairs Trading Strategy

A clean, modular implementation of a **statistical arbitrage pairs trading strategy** based on cointegration theory, applied to equity markets.

---

## Strategy Overview

Pairs trading exploits the **mean-reverting behaviour of the spread** between two cointegrated assets. When the spread deviates significantly from its historical mean, the strategy bets on a reversion to equilibrium.

### Key steps

| Step | Method |
|------|--------|
| 1. Pair selection | Engle-Granger cointegration test (p < 0.05) |
| 2. Hedge ratio | OLS regression of log prices |
| 3. Spread | `spread = log(S1) − β · log(S2)` |
| 4. Z-score | Rolling 60-day normalisation |
| 5. Signal | Entry at ±2σ, exit at ±0.5σ, stop at ±3.5σ |
| 6. Backtest | Daily P&L, Sharpe, drawdown, win rate |

### Signal logic

```
z-score < -2.0  →  LONG  spread  (buy S1, sell S2)
z-score >  2.0  →  SHORT spread  (sell S1, buy S2)
|z-score| < 0.5 →  EXIT  position
|z-score| > 3.5 →  STOP  (risk control)
```

---

## Project Structure

```
quant-pairs-trading/
├── main.py                  # Entry point — run the full pipeline
├── requirements.txt
├── src/
│   ├── data_loader.py       # Download price data via yfinance
│   ├── cointegration.py     # Engle-Granger test, spread, z-score, ADF
│   ├── strategy.py          # Signal generation and P&L computation
│   └── backtest.py          # Performance metrics and charts
└── results/                 # Output charts (auto-created)
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/m11hamedrahmani/quant-pairs-trading.git
cd quant-pairs-trading

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the strategy (default pair: GS / MS, 2018-2024)
python main.py
```

Edit the `CONFIG` block at the top of `main.py` to change the pair, dates, or thresholds.

---

## Performance Metrics

The backtest reports:

- **Total P&L** — cumulative spread return over the period
- **Annualised Sharpe Ratio** — risk-adjusted return (√252 scaling)
- **Maximum Drawdown** — largest peak-to-trough loss on the P&L curve
- **Win Rate** — percentage of profitable trading days
- **Number of trades** — long entries, short entries, stops hit

---

## Example Output (GS / MS, 2018–2024)

```
==================================================
  Pairs Trading Strategy: GS / MS
==================================================
  Total P&L            :  0.3241
  Annualised Sharpe    :  0.847
  Max Drawdown         : -0.1823
  Win Rate             :  52.4%
  Trading Days         :  1508
  Avg Daily P&L        :  0.000215
==================================================
```

---

## Limitations & Future Improvements

- **Transaction costs not modelled** — in live trading, bid-ask spreads and commissions would reduce P&L
- **Static hedge ratio** — a Kalman filter could dynamically update β over time
- **Single pair** — extending to a portfolio of cointegrated pairs would improve diversification
- **Regime detection** — adding a regime filter (e.g. VIX threshold) could avoid trading in unstable correlation periods
- **Walk-forward validation** — currently a single in-sample backtest; out-of-sample testing is needed

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `yfinance` | Price data download |
| `pandas` / `numpy` | Data manipulation |
| `statsmodels` | Cointegration test, OLS, ADF |
| `matplotlib` | Visualisation |

---

## Author

**Mohamed Rahmani**  
Bachelor in Technology & Management — ESILV (Paris La Défense)  
[LinkedIn](https://www.linkedin.com/in/mohamed-rahmani11/) · [GitHub](https://github.com/m11hamedrahmani)
