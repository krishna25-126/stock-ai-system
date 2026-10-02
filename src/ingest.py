import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import datetime as dt

UNIVERSE  = ['AAPL','MSFT','JPM','XOM','JNJ','PG','NVDA','KO','CAT','HD']
BENCHMARK = '^GSPC'
START, END = '2015-01-01', dt.date.today().isoformat()

# 1. Pull stock + benchmark + VIX prices
px = yf.download(UNIVERSE + [BENCHMARK, '^VIX'],
                  start=START, end=END, auto_adjust=False)

# 2. Pull macro data from FRED (10yr yield, 3mo yield, CPI, unemployment)
macro = web.DataReader(['DGS10','DGS3MO','CPIAUCSL','UNRATE'],
                        'fred', START, END)

# 3. Save both as parquet (compact, fast to reload)
px.to_parquet('data/raw/prices.parquet')
macro.to_parquet('data/raw/macro.parquet')

print("Prices shape:", px.shape)
print("Macro shape:", macro.shape)
print("Saved to data/raw/")