import numpy as np
import pandas as pd

def annualized_return(returns: pd.Series, periods_per_year=252) -> float:
    return returns.mean() * periods_per_year

def annualized_volatility(returns: pd.Series, periods_per_year=252) -> float:
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, risk_free=0.04, periods_per_year=252) -> float:
    excess = returns - risk_free / periods_per_year
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)

def sortino_ratio(returns: pd.Series, risk_free=0.04, periods_per_year=252) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    return (excess.mean() / downside.std()) * np.sqrt(periods_per_year)

def max_drawdown(prices: pd.Series) -> float:
    cum_max = prices.cummax()
    drawdown = (prices - cum_max) / cum_max
    return drawdown.min()

def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    cov = np.cov(asset_returns, market_returns)[0][1]
    var = np.var(market_returns)
    return cov / var


if __name__ == '__main__':
    panel = pd.read_parquet('data/processed/features.parquet')

    raw = pd.read_parquet('data/raw/prices.parquet')
    mkt_ret = np.log(raw[('Adj Close', '^GSPC')]).diff().dropna()

    print(f"{'Ticker':<8}{'AnnRet':>10}{'AnnVol':>10}{'Sharpe':>10}{'Sortino':>10}{'MaxDD':>10}{'Beta':>8}")
    universe = ['AAPL','MSFT','JPM','XOM','JNJ','PG','NVDA','KO','CAT','HD']
    for tkr in universe:
        r = panel[(tkr, 'ret1')].dropna()
        p = panel[(tkr, 'Adj Close')].dropna()
        aligned_r, aligned_mkt = r.align(mkt_ret, join='inner')

        print(f"{tkr:<8}{annualized_return(r):>10.2%}{annualized_volatility(r):>10.2%}"
              f"{sharpe_ratio(r):>10.2f}{sortino_ratio(r):>10.2f}"
              f"{max_drawdown(p):>10.2%}{beta(aligned_r, aligned_mkt):>8.2f}")