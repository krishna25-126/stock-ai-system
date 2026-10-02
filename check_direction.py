import numpy as np
y = np.load('data/processed/y_sequences.npy')
n = len(y)
i_val = int(n * 0.85)
yte = y[i_val:]
print("Test set size:", len(yte))
print("% up days in test set:", np.mean(yte > 0))
print("% down days in test set:", np.mean(yte < 0))