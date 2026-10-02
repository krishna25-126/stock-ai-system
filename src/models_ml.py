import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

FEATURE_COLS = ['ret1', 'ret5', 'p_sma10', 'p_sma20', 'p_sma50',
                 'rsi14', 'vol21', 'bb_width']


def directional_accuracy(y_true, y_pred):
    """% of days where predicted direction (up/down) matches actual direction."""
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def evaluate_model(name, model, X, y, n_splits=5):
    """Walk-forward evaluation using TimeSeriesSplit (expanding window)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, dirs = [], [], []

    for tr_idx, te_idx in tscv.split(X):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        scaler = StandardScaler().fit(Xtr)   # fit scaler on TRAIN fold only
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)

        maes.append(mean_absolute_error(yte, pred))
        rmses.append(np.sqrt(mean_squared_error(yte, pred)))
        dirs.append(directional_accuracy(yte, pred))

    print(f"{name:<18}MAE={np.mean(maes):.6f}  RMSE={np.mean(rmses):.6f}  "
          f"DirAcc={np.mean(dirs):.2%}")
    return {'model': name, 'MAE': np.mean(maes), 'RMSE': np.mean(rmses),
            'DirAcc': np.mean(dirs)}


def naive_baseline(y):
    """Naive random-walk: predict today's return will repeat tomorrow (predict 0 change)."""
    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, dirs = [], [], []
    for tr_idx, te_idx in tscv.split(y):
        yte = y[te_idx]
        pred = np.zeros_like(yte)   # naive: predict zero return (no change)
        maes.append(mean_absolute_error(yte, pred))
        rmses.append(np.sqrt(mean_squared_error(yte, pred)))
        dirs.append(directional_accuracy(yte, pred))
    print(f"{'Naive baseline':<18}MAE={np.mean(maes):.6f}  RMSE={np.mean(rmses):.6f}  "
          f"DirAcc={np.mean(dirs):.2%}")
    return {'model': 'Naive baseline', 'MAE': np.mean(maes), 'RMSE': np.mean(rmses),
            'DirAcc': np.mean(dirs)}


if __name__ == '__main__':
    panel = pd.read_parquet('data/processed/features.parquet')
    tkr = 'AAPL'

    X = panel[[(tkr, c) for c in FEATURE_COLS]].values
    y = panel[(tkr, 'ret1')].shift(-1).values  # predict next day's return

    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    print(f"Evaluating models on {tkr} — predicting next-day return\n")
    print(f"{'Model':<18}{'Metrics'}")
    print("-" * 60)

    results = []
    results.append(naive_baseline(y))
    results.append(evaluate_model('Linear Reg', LinearRegression(), X, y))
    results.append(evaluate_model('Ridge', Ridge(alpha=1.0), X, y))
    results.append(evaluate_model('Random Forest',
                    RandomForestRegressor(n_estimators=200, max_depth=5,
                                          random_state=42), X, y))
    results.append(evaluate_model('XGBoost',
                    xgb.XGBRegressor(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42), X, y))
    results.append(evaluate_model('SVR', SVR(kernel='rbf', C=1.0, epsilon=0.001), X, y))

    leaderboard = pd.DataFrame(results).sort_values('MAE')
    print("\n--- Leaderboard (sorted by MAE, lower is better) ---")
    print(leaderboard.to_string(index=False))

    leaderboard.to_csv('reports/ml_leaderboard.csv', index=False)
    print("\nSaved to reports/ml_leaderboard.csv")