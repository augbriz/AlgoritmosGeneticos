import LSTM 

hp = {
    "lstm_units": 64,
    "num_layers": 2,
    "dropout_rate": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 60
}

res = LSTM.ga_evaluate(hp, do_plots=False)
print("\n=== RESULTADO PRUEBA ===")
for k, v in res.items():
    if isinstance(v, float):
        print(f"{k}: {v:.6f}")
    else:
        print(f"{k}: {v}")
