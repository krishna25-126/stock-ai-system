import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

tf.random.set_seed(42)
np.random.seed(42)

FEATURE_COLS = ['ret1', 'ret5', 'p_sma10', 'p_sma20', 'p_sma50',
                 'rsi14', 'vol21', 'bb_width']
TICKER = 'AAPL'


def build_small_lstm(L, n_features):
    m = models.Sequential([
        layers.Input((L, n_features)),
        layers.LSTM(16),
        layers.Dropout(0.2),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mae')
    return m


def run_pilot(X, y, label):
    """Small-sample pilot: simple train/test split, no CV (too little data)."""
    n = len(X)
    split = int(n * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    scaler = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr = scaler.transform(Xtr.reshape(-1, Xtr.shape[2])).reshape(Xtr.shape)
    Xte = scaler.transform(Xte.reshape(-1, Xte.shape[2])).reshape(Xte.shape)

    model = build_small_lstm(Xtr.shape[1], Xtr.shape[2])
    es = callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_split=0.2, epochs=60,
              callbacks=[es], shuffle=False, verbose=0)

    pred = model.predict(Xte, verbose=0).flatten()
    mae = mean_absolute_error(yte, pred)
    dir_acc = np.mean(np.sign(yte) == np.sign(pred)) if len(yte) else float('nan')

    print(f"{label:<20} n_test={len(yte):<4} MAE={mae:.6f}  DirAcc={dir_acc:.2%}")
    return mae, dir_acc


if __name__ == '__main__':
    panel = pd.read_parquet('data/processed/features.parquet')
    sentiment = pd.read_parquet('data/processed/daily_sentiment.parquet')
    sent_aapl = sentiment[sentiment['ticker'] == TICKER].set_index('session_date')

    feat = panel[[(TICKER, c) for c in FEATURE_COLS]].copy()
    feat.columns = FEATURE_COLS
    feat.index = pd.to_datetime(feat.index)

    # Restrict to the overlap window where sentiment data exists
    overlap = feat.join(sent_aapl[['sentiment']], how='inner')
    overlap['target'] = panel[(TICKER, 'ret1')].reindex(overlap.index).shift(-1)
    overlap = overlap.dropna()

    print(f"Overlap window: {len(overlap)} days ({overlap.index.min().date()} to {overlap.index.max().date()})")

    if len(overlap) < 20:
        print("\nWARNING: overlap window is very small — results are illustrative only, not statistically meaningful.")

    L = min(10, max(2, len(overlap) // 3))  # short sequence length given tiny sample
    print(f"Using sequence length L={L}\n")

    def make_seq(X, y, L):
        Xs, ys = [], []
        for i in range(len(X) - L):
            Xs.append(X[i:i+L])
            ys.append(y[i+L])
        return np.array(Xs), np.array(ys)

    # WITHOUT sentiment
    X_base = overlap[FEATURE_COLS].values
    y_base = overlap['target'].values
    Xs_base, ys_base = make_seq(X_base, y_base, L)

    # WITH sentiment
    X_sent = overlap[FEATURE_COLS + ['sentiment']].values
    Xs_sent, ys_sent = make_seq(X_sent, y_base, L)