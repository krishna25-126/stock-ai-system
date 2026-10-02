\## Week 3 Finding: Directional Accuracy is Misleading

All 4 deep learning models (LSTM, GRU, BiLSTM, Transformer) achieved identical

directional accuracy (52.72%) on the test set, exactly matching the base rate

of "up" days in that window (52.72%). This indicates the models learned to

always predict a positive return rather than discriminating between up/down

days — a known failure mode on class-imbalanced test windows. MAE/RMSE still

showed modest improvement over the naive baseline, but directional accuracy

should be reported relative to this base rate, not in isolation.

