AI Stock Prediction & Portfolio Optimization

An end-to-end research pipeline that forecasts stock returns, scores market sentiment, and turns the results into a risk-optimized portfolio with transparent BUY/HOLD/SELL recommendations — wrapped in an interactive Streamlit dashboard.

Live app: https://stock-ai-system-gumnprtoggrtjyeqfwwhzg.streamlit.app

⚠️ Educational research project — not financial advice.

What it does
Pulls 10+ years of daily price data (Yahoo Finance) and macro indicators (FRED) for a 10-stock universe
Cleans and validates the data, engineers technical indicators (RSI, moving averages, volatility, Bollinger width)
Benchmarks 8 forecasting models — 4 classical ML (Linear/Ridge, Random Forest, XGBoost, SVR) and 4 deep learning (LSTM, GRU, BiLSTM, Transformer) — against a naive random-walk baseline using walk-forward (TimeSeriesSplit) evaluation
Scores financial news sentiment with FinBERT, aligned to trading sessions with no look-ahead leakage
Builds a max-Sharpe optimized portfolio (Modern Portfolio Theory via PyPortfolioOpt), cross-validated with Monte Carlo simulation
Backtests the optimized portfolio against equal-weight and S&P 500 benchmarks on a held-out period
Generates relative BUY/HOLD/SELL signals and rebalancing recommendations
Presents everything in a 5-tab Streamlit dashboard
Repo structure
stock-ai-system/
├── data/
│   ├── raw/                # downloaded, immutable (gitignored)
│   └── processed/          # cleaned panel, features, sequences
├── src/
│   ├── ingest.py           # pull prices + macro data
│   ├── clean.py            # validate, align, flag outliers
│   ├── features.py         # technical indicators
│   ├── sequences.py        # windowing for deep learning models
│   ├── finance_math.py     # Sharpe, Sortino, Beta, max drawdown (from scratch)
│   ├── gd_linreg.py        # manual gradient descent, verified vs. sklearn
│   ├── models_ml.py        # classical ML leaderboard
│   ├── models_dl.py        # deep learning leaderboard
│   ├── fetch_news.py       # pull financial headlines
│   ├── sentiment.py        # FinBERT scoring + session alignment
│   ├── sentiment_correlation.py
│   ├── portfolio.py        # max-Sharpe optimization + Monte Carlo
│   ├── backtest.py         # out-of-sample portfolio backtest
│   └── recommend.py        # BUY/HOLD/SELL + rebalancing engine
├── dashboard/app.py        # Streamlit dashboard
├── reports/                # generated CSVs, findings, written report
├── requirements.txt
├── config.yaml
└── README.md
Running it locally
bash
pip install -r requirements.txt

python src/ingest.py
python src/clean.py
python src/features.py
python src/sequences.py

python src/finance_math.py
python src/models_ml.py
python src/models_dl.py

python src/fetch_news.py          # requires NEWS_API_KEY in .env
python src/sentiment.py

python src/portfolio.py
python src/backtest.py
python src/recommend.py

streamlit run dashboard/app.py
Key findings
Classical ML and deep learning models show no meaningful improvement over a naive baseline for next-day return prediction — directional accuracy across all models sits near 50-53%, consistent with weak-form market efficiency.
A sentiment-prediction integration test was limited by news API data availability (~9-day usable overlap on the free tier); results are reported as an illustrative pilot, not a validated conclusion.
The uncapped max-Sharpe optimizer concentrated 62.7% of the portfolio in NVDA due to its outsized historical return, producing high absolute returns but a worse risk-adjusted (Sharpe) outcome than naive equal-weighting out-of-sample — a real-world reproduction of the well-documented "estimation error" critique of mean-variance optimization.

Full methodology, results, and discussion are in reports/.

Disclaimer

This project is for educational and research purposes only. Nothing it produces constitutes financial advice.
