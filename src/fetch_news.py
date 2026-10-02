import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('NEWS_API_KEY')

def fetch_headlines(query, from_date, to_date, page_size=100):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': page_size,
        'apiKey': API_KEY,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    rows = []
    for article in data.get('articles', []):
        rows.append({
            'timestamp': article['publishedAt'],
            'ticker': query,
            'text': article['title'] + '. ' + (article.get('description') or '')
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import datetime as dt
    today = dt.date.today()
    month_ago = today - dt.timedelta(days=29)  # NewsAPI free tier only allows ~1 month back

    tickers = ['AAPL', 'MSFT', 'NVDA']  # start small — free tier has rate limits
    all_news = []
    for tkr in tickers:
        df = fetch_headlines(tkr, month_ago.isoformat(), today.isoformat())
        print(f"{tkr}: {len(df)} articles fetched")
        all_news.append(df)

    news_df = pd.concat(all_news, ignore_index=True)
    news_df.to_parquet('data/raw/news.parquet')
    print(f"\nTotal: {len(news_df)} articles saved to data/raw/news.parquet")