import pandas as pd
px = pd.read_parquet('data/raw/prices.parquet')
print(px.head())
print(px['Adj Close']['AAPL'].tail())