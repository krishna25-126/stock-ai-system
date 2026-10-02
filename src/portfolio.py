import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt import plotting

UNIVERSE = ['AAPL','MSFT','JPM','XOM','JNJ','PG','NVDA','KO','CAT','HD']


def load_price_matrix():
    """Build a simple ticker-as-column price DataFrame for PyPortfolioOpt."""
    panel = pd.read_parquet('data/processed/clean_prices.parquet')
    prices = pd.DataFrame({tkr: panel[(tkr, 'Adj Close')] for tkr in UNIVERSE})
    return prices.dropna()


def optimize_portfolio(prices: pd.DataFrame, risk_free_rate=0.04):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))  # long-only, no leverage
    raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    clean_weights = ef.clean_weights()

    perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    return clean_weights, perf, mu, S


def monte_carlo_check(mu, S, n_portfolios=20000, risk_free_rate=0.04):
    """Simulate random long-only portfolios to visually confirm the efficient frontier."""
    n_assets = len(mu)
    results = np.zeros((3, n_portfolios))

    for i in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= w.sum()
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w.T, np.dot(S, w)))
        sharpe = (port_ret - risk_free_rate) / port_vol
        results[:, i] = [port_ret, port_vol, sharpe]

    best_idx = np.argmax(results[2])
    return {
        'best_simulated_return': results[0, best_idx],
        'best_simulated_vol': results[1, best_idx],
        'best_simulated_sharpe': results[2, best_idx],
    }


if __name__ == '__main__':
    prices = load_price_matrix()
    print(f"Price matrix shape: {prices.shape} (dates x tickers)\n")

    weights, perf, mu, S = optimize_portfolio(prices)

    print("--- Optimal (Max Sharpe) Portfolio Weights ---")
    for tkr, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w > 0.001:
            print(f"  {tkr:<6} {w:.1%}")

    print(f"\nExpected annual return: {perf[0]:.2%}")
    print(f"Annual volatility:      {perf[1]:.2%}")
    print(f"Sharpe ratio:           {perf[2]:.2f}")

    print("\n--- Monte Carlo cross-check (20,000 random portfolios) ---")
    mc = monte_carlo_check(mu, S)
    print(f"Best simulated return: {mc['best_simulated_return']:.2%}")
    print(f"Best simulated vol:    {mc['best_simulated_vol']:.2%}")
    print(f"Best simulated Sharpe: {mc['best_simulated_sharpe']:.2f}")
    print("\n(Analytical and simulated Sharpe should be close — simulation can't")
    print(" exactly reach the analytical optimum but should approach it.)")

    pd.Series(weights).to_csv('reports/optimal_weights.csv')
    print("\nSaved weights to reports/optimal_weights.csv")