import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator features to a single ticker's cleaned OHLCV frame."""
    df = df.copy()
    p = df['Adj Close']

    # Returns
    df['ret1'] = np.log(p).diff()
    df['ret5'] = np.log(p).diff(5)

    # Moving averages + price-relative-to-MA
    for w in (10, 20, 50, 200):
        df[f'sma{w}']   = p.rolling(w).mean()
        df[f'p_sma{w}'] = p / df[f'sma{w}']

    # RSI (14-day)
    d = p.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.rolling(14).mean() / dn.rolling(14).mean()
    df['rsi14'] = 100 - 100 / (1 + rs)

    # Volatility (21-day rolling std of returns)
    df['vol21'] = df['ret1'].rolling(21).std()

    # Bollinger band width (20-day)
    mid = p.rolling(20).mean()
    sd  = p.rolling(20).std()
    df['bb_width'] = (4 * sd) / mid

    # Calendar features
    df['dow']   = df.index.dayofweek
    df['month'] = df.index.month

    # Drop rows with NaNs from the rolling-window warm-up period
    return df.dropna()


if __name__ == '__main__':
    panel = pd.read_parquet('data/processed/clean_prices.parquet')
    universe = ['AAPL','MSFT','JPM','XOM','JNJ','PG','NVDA','KO','CAT','HD']

    featured = {}
    for tkr in universe:
        t = panel[tkr].copy()
        t = add_features(t)
        featured[tkr] = t
        print(f"{tkr}: {len(t)} rows after feature engineering, {t.shape[1]} columns")

    out = pd.concat(featured, axis=1)
    out.to_parquet('data/processed/features.parquet')
    print("\nSaved to data/processed/features.parquet")
    print("Final shape:", out.shape)