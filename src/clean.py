import numpy as np
import pandas as pd

def clean_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Clean OHLCV data for a single ticker from the wide multi-index frame."""
    cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    t = df[[(c, ticker) for c in cols]].copy()
    t.columns = cols

    # Forward-fill short gaps only (max 3 trading days) — never back-fill
    t[['Open','High','Low','Close','Adj Close']] = (
        t[['Open','High','Low','Close','Adj Close']].ffill(limit=3))
    t['Volume'] = t['Volume'].fillna(0)

    # Flag OHLC invariant violations (Low should be the min, High the max)
    bad = ((t['Low'] > t[['Open','Close','High']].min(axis=1)) |
           (t['High'] < t[['Open','Close','Low']].max(axis=1)))
    t.loc[bad, :] = np.nan

    # Flag statistical outliers in returns (don't delete, just flag)
    r = np.log(t['Adj Close']).diff()
    z = (r - r.rolling(63).mean()) / r.rolling(63).std()
    t['is_outlier'] = z.abs() > 5

    return t.dropna(subset=['Adj Close'])


if __name__ == '__main__':
    px = pd.read_parquet('data/raw/prices.parquet')
    universe = ['AAPL','MSFT','JPM','XOM','JNJ','PG','NVDA','KO','CAT','HD']

    cleaned = {}
    for tkr in universe:
        c = clean_prices(px, tkr)
        cleaned[tkr] = c
        print(f"{tkr}: {len(c)} clean rows, {c['is_outlier'].sum()} outliers flagged")

    # Save each ticker's cleaned data, plus a combined panel
    panel = pd.concat(cleaned, axis=1)
    panel.to_parquet('data/processed/clean_prices.parquet')
    print("\nSaved combined panel to data/processed/clean_prices.parquet")
    print("Panel shape:", panel.shape)