import numpy as np
import pandas as pd

UNIVERSE = ['AAPL','MSFT','JPM','XOM','JNJ','PG','NVDA','KO','CAT','HD']


def composite_score(forecast_return, sentiment, volatility,
                     w_forecast=0.5, w_sentiment=0.3, w_risk=0.2):
    """
    Combine forecast signal, sentiment, and risk into one score.
    Higher score = more bullish. Volatility penalizes the score (risk-adjusted).
    """
    norm_vol = volatility / volatility.max() if volatility.max() > 0 else volatility
    score = (w_forecast * forecast_return
             + w_sentiment * sentiment
             - w_risk * norm_vol)
    return score


def recommend(score, low_thresh, high_thresh):
    if score >= high_thresh:
        return 'BUY'
    elif score <= low_thresh:
        return 'SELL'
    else:
        return 'HOLD'


def rebalance_actions(current_weights: dict, target_weights: dict, band=0.05):
    """Flag any asset whose weight has drifted more than `band` from target."""
    actions = {}
    all_assets = set(current_weights) | set(target_weights)
    for asset in all_assets:
        cur = current_weights.get(asset, 0)
        tgt = target_weights.get(asset, 0)
        drift = tgt - cur
        if abs(drift) > band:
            direction = 'INCREASE' if drift > 0 else 'REDUCE'
            actions[asset] = {'current': round(cur, 3), 'target': round(tgt, 3),
                               'action': direction, 'drift': round(drift, 3)}
    return actions


if __name__ == '__main__':
    panel = pd.read_parquet('data/processed/features.parquet')
    sentiment = pd.read_parquet('data/processed/daily_sentiment.parquet')
    optimal_weights = pd.read_csv('reports/optimal_weights.csv', index_col=0).iloc[:, 0].to_dict()

    rows = []
    for tkr in UNIVERSE:
        # Use most recent 21-day average return as a simple "forecast" proxy
        # (swap this for your actual LSTM/BiLSTM predictions once wired in)
        recent_ret = panel[(tkr, 'ret1')].dropna().tail(21).mean()
        vol = panel[(tkr, 'vol21')].dropna().iloc[-1]

        tkr_sent = sentiment[sentiment['ticker'] == tkr]
        sent_score = tkr_sent['sentiment'].tail(5).mean() if len(tkr_sent) else 0.0
        sent_score = 0.0 if pd.isna(sent_score) else sent_score

        rows.append({'ticker': tkr, 'forecast_return': recent_ret,
                     'sentiment': sent_score, 'volatility': vol})

    df = pd.DataFrame(rows).set_index('ticker')
    df['score'] = composite_score(df['forecast_return'], df['sentiment'], df['volatility'])

    # Thresholds set at the 33rd/67th percentile of today's scores (adaptive, not fixed)
    low_thresh = df['score'].quantile(0.33)
    high_thresh = df['score'].quantile(0.67)
    df['recommendation'] = df['score'].apply(lambda s: recommend(s, low_thresh, high_thresh))

    print("--- Recommendations ---")
    print(df.sort_values('score', ascending=False).to_string())

    print(f"\n(Thresholds — BUY above {high_thresh:.5f}, SELL below {low_thresh:.5f})")

    print("\n--- Rebalancing vs. Optimal Weights ---")
    # Example: assume currently equal-weighted, compare to optimal target
    current_weights = {tkr: 1 / len(UNIVERSE) for tkr in UNIVERSE}
    actions = rebalance_actions(current_weights, optimal_weights)
    if actions:
        for asset, info in sorted(actions.items(), key=lambda x: -abs(x[1]['drift'])):
            print(f"  {asset:<6} {info['action']:<10} current={info['current']:.1%} "
                  f"-> target={info['target']:.1%} (drift={info['drift']:+.1%})")
    else:
        print("  No rebalancing needed — within band.")

    df.to_csv('reports/recommendations.csv')
    print("\nSaved to reports/recommendations.csv")
