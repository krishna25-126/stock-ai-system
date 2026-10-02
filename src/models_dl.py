import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

tf.random.set_seed(42)
np.random.seed(42)


def chrono_split(X, y, train=0.70, val=0.15):
    """Chronological 70/15/15 split — never shuffle time series."""
    n = len(X)
    i_train = int(n * train)
    i_val = int(n * (train + val))
    return (X[:i_train], y[:i_train],
            X[i_train:i_val], y[i_train:i_val],
            X[i_val:], y[i_val:])


def scale_sequences(Xtr, Xval, Xte):
    """Fit a scaler on flattened TRAIN data only, apply to all three sets."""
    n_feat = Xtr.shape[2]
    scaler = StandardScaler().fit(Xtr.reshape(-1, n_feat))

    def apply(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, n_feat)).reshape(shape)

    return apply(Xtr), apply(Xval), apply(Xte)


def build_lstm(L, n_features):
    m = models.Sequential([
        layers.Input((L, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='huber', metrics=['mae'])
    return m


def build_gru(L, n_features):
    m = models.Sequential([
        layers.Input((L, n_features)),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='huber', metrics=['mae'])
    return m


def build_bilstm(L, n_features):
    m = models.Sequential([
        layers.Input((L, n_features)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='huber', metrics=['mae'])
    return m


def transformer_block(x, heads=4, key_dim=32, ff=64, drop=0.1):
    a = layers.MultiHeadAttention(heads, key_dim)(x, x)
    a = layers.Dropout(drop)(a)
    x = layers.LayerNormalization()(x + a)
    f = layers.Dense(ff, activation='relu')(x)
    f = layers.Dense(x.shape[-1])(f)
    return layers.LayerNormalization()(x + f)


def build_transformer(L, n_features):
    inp = layers.Input((L, n_features))
    x = layers.Dense(32)(inp)   # project features to embedding dim
    x = transformer_block(x)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='huber', metrics=['mae'])
    return m


def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred.flatten()))


def train_and_eval(name, build_fn, Xtr, ytr, Xval, yval, Xte, yte):
    print(f"\nTraining {name}...")
    model = build_fn(Xtr.shape[1], Xtr.shape[2])
    es = callbacks.EarlyStopping(patience=12, restore_best_weights=True)

    model.fit(Xtr, ytr, validation_data=(Xval, yval),
              epochs=100, batch_size=64, callbacks=[es],
              shuffle=False, verbose=0)

    pred = model.predict(Xte, verbose=0)
    mae = mean_absolute_error(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    dir_acc = directional_accuracy(yte, pred)

    print(f"{name:<14}MAE={mae:.6f}  RMSE={rmse:.6f}  DirAcc={dir_acc:.2%}")
    model.save(f'models/{name.lower()}.keras')
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'DirAcc': dir_acc}


if __name__ == '__main__':
    X = np.load('data/processed/X_sequences.npy')
    y = np.load('data/processed/y_sequences.npy')

    Xtr, ytr, Xval, yval, Xte, yte = chrono_split(X, y)
    Xtr, Xval, Xte = scale_sequences(Xtr, Xval, Xte)

    print(f"Train: {Xtr.shape}  Val: {Xval.shape}  Test: {Xte.shape}")

    results = []
    results.append(train_and_eval('LSTM', build_lstm, Xtr, ytr, Xval, yval, Xte, yte))
    results.append(train_and_eval('GRU', build_gru, Xtr, ytr, Xval, yval, Xte, yte))
    results.append(train_and_eval('BiLSTM', build_bilstm, Xtr, ytr, Xval, yval, Xte, yte))
    results.append(train_and_eval('Transformer', build_transformer, Xtr, ytr, Xval, yval, Xte, yte))

    import pandas as pd
    leaderboard = pd.DataFrame(results).sort_values('MAE')
    print("\n--- Deep Learning Leaderboard ---")
    print(leaderboard.to_string(index=False))
    leaderboard.to_csv('reports/dl_leaderboard.csv', index=False)