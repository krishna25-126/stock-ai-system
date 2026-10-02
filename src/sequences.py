import numpy as np
import pandas as pd

FEATURE_COLS = ['ret1', 'ret5', 'p_sma10', 'p_sma20', 'p_sma50',
                 'rsi14', 'vol21', 'bb_width']

def make_sequences(X: np.ndarray, y: np.ndarray, L=60, horizon=1):
    """Turn a flat feature table into overlapping sequences of length L."""
    Xs, ys = [], []
    for i in range(len(X) - L - horizon + 1):
        Xs.append(X[i:i+L])
        ys.append(y[i+L+horizon-1])
    return np.array(Xs), np.array(ys)


if __name__ == '__main__':
    panel = pd.read_parquet('data/processed/features.parquet')
    tkr = 'AAPL'

    X = panel[[(tkr, c) for c in FEATURE_COLS]].values
    y = panel[(tkr, 'ret1')].shift(-1).values

    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    Xs, ys = make_sequences(X, y, L=60, horizon=1)
    print("Sequence shape (samples, timesteps, features):", Xs.shape)
    print("Target shape:", ys.shape)

    np.save('data/processed/X_sequences.npy', Xs)
    np.save('data/processed/y_sequences.npy', ys)
    print("Saved sequences to data/processed/")