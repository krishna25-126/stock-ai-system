import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models

UNIVERSE = ['AAPL','MSFT','JPM','XOM','JNJ','PG','NVDA','KO','CAT','HD']


def load_price_matrix():
    panel = pd.read_parquet('data/processed/clean_prices.parquet')
    prices = pd.DataFrame({tkr: panel[(tkr, 'Adj Close')] for tkr in UNIVERSE})
    return prices.dropna()


def load_benchmark():
    raw = pd.read_parquet('data/raw/prices.parquet')
    bench = raw[('Adj Close', '^GSPC')].dropna()
    return bench


def backtest_portfolio(returns: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights).reindex(returns.columns).fillna(0)
    port_returns = returns @ w
    return (1 + port_returns).cumprod()


if __name__ == '__main__':
    prices = load_price_matrix()
    bench = load_benchmark()

    # IMPORTANT: split chronologically — optimize on the FIRST 80%, test on the LAST 20%
    split_idx = int(len(prices) * 0.80)
    train_prices = prices.iloc[:split_idx]
    test_prices = prices.iloc[split_idx:]

    print(f"Train period: {train_prices.index.min().date()} to {train_prices.index.max().date()}")
    print(f"Test period:  {test_prices.index.min().date()} to {test_prices.index.max().date()}\n")

    # Optimize using ONLY the train period (no look-ahead)
    mu = expected_returns.mean_historical_return(train_prices)
    S = risk_models.CovarianceShrinkage(train_prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef.max_sharpe(risk_free_rate=0.04)
    optimal_weights = ef.clean_weights()

    equal_weights = {tkr: 1 / len(UNIVERSE) for tkr in UNIVERSE}

    # Backtest on the TEST period only
    test_returns = test_prices.pct_change().dropna()

    optimal_curve = backtest_portfolio(test_returns, optimal_weights)
    equal_curve = backtest_portfolio(test_returns, equal_weights)

    bench_test = bench.reindex(test_prices.index).dropna()
    bench_returns = bench_test.pct_change().dropna()
    bench_curve = (1 + bench_returns).cumprod()

    def summarize(curve, name, returns_series):
        total_return = curve.iloc[-1] - 1
        ann_vol = returns_series.std() * np.sqrt(252)
        sharpe = (returns_series.mean() * 252 - 0.04) / ann_vol
        max_dd = ((curve - curve.cummax()) / curve.cummax()).min()
        print(f"{name:<20} TotalRet={total_return:>8.2%}  AnnVol={ann_vol:>8.2%}  "
              f"Sharpe={sharpe:>6.2f}  MaxDD={max_dd:>8.2%}")

    print("--- Out-of-Sample Backtest Results ---")
    summarize(optimal_curve, 'Optimized Portfolio', test_returns @ pd.Series(optimal_weights))
    summarize(equal_curve, 'Equal-Weight', test_returns @ pd.Series(equal_weights))
    summarize(bench_curve, 'S&P 500 Benchmark', bench_returns)

    results = pd.DataFrame({
        'Optimized': optimal_curve,
        'EqualWeight': equal_curve,
        'Benchmark': bench_curve
    })
    results.to_csv('reports/backtest_curves.csv')
    print("\nSaved growth curves to reports/backtest_curves.csv")