# === GA MINIMO: generación 0 (random search + ranking) ===

import os, random, numpy as np
# Reproducibilidad 
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
random.seed(42); np.random.seed(42)

import LSTM

# ----- Espacio de búsqueda (pequeño para empezar) -----
SPACE = {
    "lstm_units":   (32, 96),             # enteros
    "num_layers":   (1, 3),               # enteros
    "dropout_rate": (0.05, 0.35),         # float
    "learning_rate":(1e-4, 3e-3),         # float 
    "batch_size":   [16, 32, 64],         # categórico
    "epochs":       (40, 90),             # enteros 
    # Nota: PREDICTION_HORIZON queda fijo en 5 (lo maneja lstm.py)
    # Nota: lookback_window lo mantengo fijo al inicio para no rearmar X,y en este paso
}

POP_SIZE = 8   # Tamaño de la población -> población chica para probar
DO_PLOTS = False  # no graficar durante la evaluación para ir rápido

def sample_hp():
    """Muestrea un individuo (set de hiperparámetros) sobre el espacio de busqueda."""
    def randint(lo, hi): return int(np.random.randint(lo, hi+1))
    def randfloat(lo, hi): return float(np.random.uniform(lo, hi))
    def choice(lst): return random.choice(lst)

    hp = {
        "lstm_units":    randint(*SPACE["lstm_units"]),
        "num_layers":    randint(*SPACE["num_layers"]),
        "dropout_rate":  randfloat(*SPACE["dropout_rate"]),
        "learning_rate": 10 ** np.random.uniform(np.log10(SPACE["learning_rate"][0]),
                                                 np.log10(SPACE["learning_rate"][1])),
        "batch_size":    choice(SPACE["batch_size"]),
        "epochs":        randint(*SPACE["epochs"]),
    }
    # Redondear dropout a 3 decimales (opcional, por prolijidad)
    hp["dropout_rate"] = round(hp["dropout_rate"], 3)
    # Redondear LR
    hp["learning_rate"] = float(f"{hp['learning_rate']:.6f}")
    return hp

def evaluate_individual(hp, idx=None):
    """Evalúa un individuo llamando a lstm.ga_evaluate(hp)."""
    tag = f"[ind {idx}]" if idx is not None else ""
    print(f"{tag} Evaluando HP: {hp}")
    res = LSTM.ga_evaluate(hp, do_plots=DO_PLOTS)
    print(f"{tag} -> fitness={res['fitness']:.6f}  rmse_usd={res['rmse_prom_usd']:.3f}  best_epoch={res['best_epoch']}")
    return res

def rank_population(results):
    """Ordena por fitness ascendente (menor es mejor)."""
    return sorted(results, key=lambda r: r["fitness"])

def main():
    # 1) Crear población inicial
    population = [sample_hp() for _ in range(POP_SIZE)]

    # 2) Evaluar población
    results = []
    for i, hp in enumerate(population, 1):
        res = evaluate_individual(hp, idx=i)
        results.append(res)

    # 3) Rankear y mostrar top-k
    ranked = rank_population(results)
    k = min(5, len(ranked))
    print("\n=== TOP resultados (generación 0) ===")
    for i, r in enumerate(ranked[:k], 1):
        p = r["params"]
        print(f"#{i}  fitness={r['fitness']:.6f}  rmse_usd={r['rmse_prom_usd']:.3f}  "
              f"units={p['lstm_units']} layers={p['num_layers']} drop={p['dropout_rate']:.3f} "
              f"lr={p['learning_rate']:.6f} bs={p['batch_size']} ep={p['epochs']}")

    # 4) Guardar el mejor para referencia
    best = ranked[0]
    print("\n=== MEJOR INDIVIDUO ===")
    print(best)

if __name__ == "__main__":
    main()
