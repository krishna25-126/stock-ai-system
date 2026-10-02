import pandas as pd
import numpy as np

FEATURE_COLS = ['ret1']
TICKER = 'AAPL'

panel = pd.read_parquet('data/processed/features.parquet')
sentiment = pd.read_parquet('data/processed/daily_sentiment.parquet')
sent_aapl = sentiment[sentiment['ticker'] == TICKER].set_index('session_date')

feat = panel[[(TICKER, c) for c in FEATURE_COLS]].copy()
feat.columns = FEATURE_COLS
feat.index = pd.to_datetime(feat.index)

overlap = feat.join(sent_aapl[['sentiment', 'n_articles']], how='inner')
overlap['next_day_return'] = panel[(TICKER, 'ret1')].reindex(overlap.index).shift(-1)
overlap = overlap.dropna()

print(f"Overlap sample: {len(overlap)} days\n")
print(overlap[['sentiment', 'n_articles', 'next_day_return']])

corr = overlap['sentiment'].corr(overlap['next_day_return'])
dir_match = np.mean(np.sign(overlap['sentiment']) == np.sign(overlap['next_day_return']))

print(f"\nCorrelation (sentiment vs next-day return): {corr:.3f}")
print(f"Direction match rate: {dir_match:.2%}")
print(f"\nNOTE: n={len(overlap)} — far too small for statistical significance.")
print("This is a descriptive pilot observation only, not a validated predictive result.")