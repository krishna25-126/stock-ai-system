import pandas as pd
from transformers import pipeline

def score_headlines(news_df: pd.DataFrame) -> pd.DataFrame:
    """Score headlines with FinBERT and aggregate to daily sentiment per ticker."""
    print("Loading FinBERT model (first run downloads ~400MB)...")
    finbert = pipeline('sentiment-analysis', model='ProsusAI/finbert', truncation=True)

    print(f"Scoring {len(news_df)} headlines...")
    results = finbert(news_df['text'].tolist())

    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    news_df = news_df.copy()
    news_df['label'] = [r['label'] for r in results]
    news_df['polarity'] = [label_map[r['label']] * r['score'] for r in results]

    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])

    # IMPORTANT: news published after market close (4pm ET) should count toward
    # the NEXT trading session, not the one that already happened.
    news_df['session_date'] = news_df['timestamp'].dt.tz_localize(None).dt.normalize()
    after_close = news_df['timestamp'].dt.tz_localize(None).dt.hour >= 16
    news_df.loc[after_close, 'session_date'] += pd.Timedelta(days=1)

    daily = (news_df.groupby(['ticker', 'session_date'])['polarity']
                     .agg(['mean', 'count'])
                     .rename(columns={'mean': 'sentiment', 'count': 'n_articles'})
                     .reset_index())
    return daily


if __name__ == '__main__':
    news = pd.read_parquet('data/raw/news.parquet')
    daily_sentiment = score_headlines(news)

    print("\nSample output:")
    print(daily_sentiment.head(10))

    daily_sentiment.to_parquet('data/processed/daily_sentiment.parquet')
    print(f"\nSaved {len(daily_sentiment)} ticker-day sentiment rows to data/processed/daily_sentiment.parquet")