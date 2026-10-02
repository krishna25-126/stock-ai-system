import pandas as pd

panel = pd.read_parquet('data/processed/features.parquet')
sentiment = pd.read_parquet('data/processed/daily_sentiment.parquet')

feat_dates = pd.to_datetime(panel.index)
print("Features date range:", feat_dates.min().date(), "to", feat_dates.max().date())

sent_aapl = sentiment[sentiment['ticker'] == 'AAPL']
print("Sentiment date range:", sent_aapl['session_date'].min().date(), "to", sent_aapl['session_date'].max().date())
print("Sentiment dates available:", sorted(sent_aapl['session_date'].dt.date.unique()))