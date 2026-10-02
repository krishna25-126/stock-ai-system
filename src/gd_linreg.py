import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def gd_linreg(X: np.ndarray, y: np.ndarray, lr=1e-2, epochs=2000):
    """Linear regression trained by hand with gradient descent."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0

    for epoch in range(epochs):
        yhat = X @ w + b
        err = yhat - y

        # Gradients of mean squared error w.r.t. w and b
        grad_w = (2 / n) * (X.T @ err)
        grad_b = (2 / n) * err.sum()

        w -= lr * grad_w
        b -= lr * grad_b

        if epoch % 400 == 0:
            mse = np.mean(err ** 2)
            print(f"  epoch {epoch:4d} | MSE = {mse:.6f}")

    return w, b


if __name__ == '__main__':
    panel = pd.read_parquet('data/processed/features.parquet')
    tkr = 'AAPL'

    # Predict next-day return using today's features
    feature_cols = ['ret1', 'ret5', 'p_sma10', 'p_sma20', 'rsi14', 'vol21', 'bb_width']
    X = panel[[(tkr, c) for c in feature_cols]].values
    y = panel[(tkr, 'ret1')].shift(-1).values  # next day's return

    # Drop the last row (no next-day target) and any NaNs
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    # Scale features — gradient descent converges much faster on scaled data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training manual gradient descent model...")
    w_manual, b_manual = gd_linreg(X_scaled, y, lr=0.01, epochs=2000)

    print("\nTraining scikit-learn LinearRegression (closed-form OLS)...")
    sk_model = LinearRegression()
    sk_model.fit(X_scaled, y)

    print("\n--- Comparison ---")
    print("Manual weights:  ", np.round(w_manual, 5))
    print("Sklearn weights: ", np.round(sk_model.coef_, 5))
    print("Manual bias:     ", round(b_manual, 5))
    print("Sklearn bias:    ", round(sk_model.intercept_, 5))

    manual_pred = X_scaled @ w_manual + b_manual
    sk_pred = sk_model.predict(X_scaled)
    print("\nManual MSE:  ", np.mean((manual_pred - y) ** 2))
    print("Sklearn MSE: ", np.mean((sk_pred - y) ** 2))