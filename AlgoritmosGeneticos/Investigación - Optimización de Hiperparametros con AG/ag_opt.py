import os, random, numpy as np
# Reproducibilidad 
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
random.seed(42); np.random.seed(42)

import LSTM
import matplotlib.pyplot as plt

# ----- Espacio de búsqueda  
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

# ----- Hiperparámetros del GA -----
POP_SIZE = 8   # Tamaño de la población -> población chica para probar
N_GEN      = 2 # generaciones
ELITE_K    = 2 # elitismo (cuantos pasan directo)
TOURN_K    = 3 # torneo 
CX_RATE    = 0.9 # prob. crossover
MUT_RATE   = 0.3 # prob. mutación

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
    # Redondear dropout a 3 decimales 
    hp["dropout_rate"] = round(hp["dropout_rate"], 3)
    # Redondear LR
    hp["learning_rate"] = float(f"{hp['learning_rate']:.6f}")
    return hp

def evaluate_individual(hp, idx=None, gen=None):
    """Evalúa un individuo llamando a lstm.ga_evaluate(hp)."""
    gen_tag = f"Gen {gen} " if gen is not None else ""
    idx_tag = f"[ind {idx}]" if idx is not None else ""
    print(f"{gen_tag}{idx_tag} Evaluando HP: {hp}")
    res = LSTM.ga_evaluate(hp, do_plots=DO_PLOTS)
    print(f"{gen_tag}{idx_tag} -> fitness={res['fitness']:.6f}  rmse_usd={res['rmse_prom_usd']:.3f}  best_epoch={res['best_epoch']}")
    return res

def rank_population(results):
    """Ordena por fitness ascendente (menor es mejor)."""
    return sorted(results, key=lambda r: r["fitness"])

def rank_population(results):
    """Ordena por fitness (menor es mejor)."""
    return sorted(results, key=lambda r: r["fitness"])

def tournament_select(results, k=TOURN_K):
    """Devuelve los HP del mejor entre k individuos al azar (selección por torneo)."""
    import numpy as np
    idxs = np.random.choice(len(results), size=k, replace=False)
    winner = min((results[i] for i in idxs), key=lambda r: r["fitness"])
    # devolvemos SOLO los hiperparámetros
    return dict(winner["params"])

def _bound_int(x, lo, hi):
    return int(min(max(int(x), lo), hi))

def _bound_float(x, lo, hi):
    return float(min(max(float(x), lo), hi))

def crossover(hp1, hp2):
    """
    Cruce simple:
    - enteros/categóricos: intercambio 50/50
    - floats: promedio + pequeño ruido (ya acotado)
    Devuelve dos hijos dict.
    """
    import numpy as np
    child1, child2 = dict(hp1), dict(hp2)

    # ----- enteros (swap 50/50) -----
    for key, (lo, hi) in [("lstm_units", SPACE["lstm_units"]),
                          ("num_layers", SPACE["num_layers"]),
                          ("epochs", SPACE["epochs"])]:
        if np.random.rand() < 0.5:
            child1[key], child2[key] = child2[key], child1[key]
        # asegurar límites (por si venimos “sucios” de generaciones previas)
        child1[key] = _bound_int(child1[key], lo, hi)
        child2[key] = _bound_int(child2[key], lo, hi)

    # ----- categórico (swap 50/50) -----
    if np.random.rand() < 0.5:
        child1["batch_size"], child2["batch_size"] = child2["batch_size"], child1["batch_size"]
    # si por algún motivo se salió, volvemos al más cercano permitido
    if child1["batch_size"] not in SPACE["batch_size"]:
        child1["batch_size"] = min(SPACE["batch_size"], key=lambda x: abs(x - child1["batch_size"]))
    if child2["batch_size"] not in SPACE["batch_size"]:
        child2["batch_size"] = min(SPACE["batch_size"], key=lambda x: abs(x - child2["batch_size"]))

    # ----- floats (promedio + ruido Gaussian) -----
    # dropout
    if np.random.rand() < 0.5:
        m = (hp1["dropout_rate"] + hp2["dropout_rate"]) / 2.0
        child1["dropout_rate"] = _bound_float(m + np.random.normal(0, 0.02), *SPACE["dropout_rate"])
        child2["dropout_rate"] = _bound_float(m + np.random.normal(0, 0.02), *SPACE["dropout_rate"])

    # learning rate (mezcla en log-escala: más estable)
    if np.random.rand() < 0.5:
        lo, hi = SPACE["learning_rate"]
        logm = (np.log10(hp1["learning_rate"]) + np.log10(hp2["learning_rate"])) / 2.0
        child1["learning_rate"] = _bound_float(10 ** (logm + np.random.normal(0, 0.15)), lo, hi)
        child2["learning_rate"] = _bound_float(10 ** (logm + np.random.normal(0, 0.15)), lo, hi)

    return child1, child2

def mutate(hp, p=MUT_RATE):
    """Cada gen tiene probabilidad p de mutar. Siempre respeta límites."""
    import numpy as np
    h = dict(hp)

    # enteros
    if np.random.rand() < p:
        lo, hi = SPACE["lstm_units"]
        h["lstm_units"] = _bound_int(h["lstm_units"] + np.random.choice([-8, -4, 4, 8]), lo, hi)

    if np.random.rand() < p:
        lo, hi = SPACE["num_layers"]
        h["num_layers"] = _bound_int(h["num_layers"] + np.random.choice([-1, 1]), lo, hi)

    if np.random.rand() < p:
        lo, hi = SPACE["epochs"]
        h["epochs"] = _bound_int(h["epochs"] + np.random.choice([-10, -5, 5, 10]), lo, hi)

    # categórico -> mover al vecino si existe (o re-sample)
    if np.random.rand() < p:
        choices = SPACE["batch_size"]
        if h["batch_size"] in choices:
            i = choices.index(h["batch_size"])
            if i == 0:
                h["batch_size"] = choices[1]
            elif i == len(choices)-1:
                h["batch_size"] = choices[-2]
            else:
                h["batch_size"] = choices[i + np.random.choice([-1, 1])]
        else:
            h["batch_size"] = random.choice(choices)

    # floats
    if np.random.rand() < p:
        lo, hi = SPACE["dropout_rate"]
        h["dropout_rate"] = _bound_float(h["dropout_rate"] + np.random.normal(0, 0.03), lo, hi)

    if np.random.rand() < p:
        lo, hi = SPACE["learning_rate"]
        h["learning_rate"] = _bound_float(h["learning_rate"] * (10 ** np.random.normal(0, 0.15)), lo, hi)

    return h


def mostrar_distribucion_rmse(todos_los_rmse):
    """Muestra un histograma simple de la distribución de RMSE obtenidos."""
    plt.figure(figsize=(10, 6))
    plt.hist(todos_los_rmse, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribución de RMSE (USD) - Todas las Generaciones', fontsize=14, fontweight='bold')
    plt.xlabel('RMSE (USD)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    
    # Estadísticas básicas
    media = np.mean(todos_los_rmse)
    mediana = np.median(todos_los_rmse)
    mejor = min(todos_los_rmse)
    peor = max(todos_los_rmse)
    
    # Líneas verticales para estadísticas
    plt.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: ${media:.3f}')
    plt.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: ${mediana:.3f}')
    plt.axvline(mejor, color='blue', linestyle='-', linewidth=2, label=f'Mejor: ${mejor:.3f}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Resumen en consola
    print(f"\n{'='*50}")
    print(f"RESUMEN DE DISTRIBUCIÓN DE RMSE")
    print(f"{'='*50}")
    print(f" Total de evaluaciones: {len(todos_los_rmse)}")
    print(f" Mejor RMSE: ${mejor:.3f} USD")
    print(f" Peor RMSE: ${peor:.3f} USD")
    print(f" Media: ${media:.3f} USD")
    print(f" Mediana: ${mediana:.3f} USD")
    print(f" Desviación estándar: ${np.std(todos_los_rmse):.3f} USD")
    print(f"{'='*50}")


def main():
    # Lista para recolectar todos los RMSE
    todos_los_rmse = []
    # 1) Crear población inicial
    population = [sample_hp() for _ in range(POP_SIZE)]

    # 2) Evaluar población
    results = []
    for i, hp in enumerate(population, 1):
        res = evaluate_individual(hp, idx=i, gen=1)
        results.append(res)
        todos_los_rmse.append(res['rmse_prom_usd'])  # Recolectar RMSE

    # 3) Rankear y mostrar top-k
    ranked = rank_population(results)
    # === Generaciones 1..N_GEN ===
    parents = ranked  # mejores de G0 como base
    for gen in range(1, N_GEN + 1):
        print(f"\n=== Generación {gen} ===")

    # 1) Elitismo: copiamos los mejores ELITE_K
    next_population = [dict(parents[i]["params"]) for i in range(min(ELITE_K, len(parents)))]

    # 2) Rellenar con descendencia (selección -> cruce -> mutación)
    while len(next_population) < POP_SIZE:
        p1 = tournament_select(parents, k=TOURN_K)
        p2 = tournament_select(parents, k=TOURN_K)

        # cruce
        if np.random.rand() < CX_RATE:
            c1, c2 = crossover(p1, p2)
        else:
            c1, c2 = dict(p1), dict(p2)

        # mutación (acotada)
        c1 = mutate(c1, p=MUT_RATE)
        c2 = mutate(c2, p=MUT_RATE)

        next_population.append(c1)
        if len(next_population) < POP_SIZE:
            next_population.append(c2)

    # 3) Evaluar nueva población
    pop_results = []
    for i, hp in enumerate(next_population, 1):
        res = evaluate_individual(hp, idx=i, gen=gen)
        pop_results.append(res)
        todos_los_rmse.append(res['rmse_prom_usd'])  # Recolectar RMSE

    # 4) Rankear para la siguiente iteración
    parents = rank_population(pop_results)

    # 5) Reporte
    print(f"\n--- TOP G{gen} ---")
    for i, r in enumerate(parents[:min(5, len(parents))], 1):
        p = r["params"]
        print(f"#{i} fitness={r['fitness']:.6f} rmse_usd={r['rmse_prom_usd']:.3f} "
              f"units={p['lstm_units']} layers={p['num_layers']} drop={p['dropout_rate']:.3f} "
              f"lr={p['learning_rate']:.6f} bs={p['batch_size']} ep={p['epochs']}")

    k = min(5, len(ranked))
    print("\n=== TOP resultados (generación 0) ===")
    for i, r in enumerate(ranked[:k], 1):
        p = r["params"]
        print(f"#{i}  fitness={r['fitness']:.6f}  rmse_usd={r['rmse_prom_usd']:.3f}  "
              f"units={p['lstm_units']} layers={p['num_layers']} drop={p['dropout_rate']:.3f} "
              f"lr={p['learning_rate']:.6f} bs={p['batch_size']} ep={p['epochs']}")

    # 4) Guardar el mejor para referencia
    best = parents[0]
    print("\n=== MEJOR INDIVIDUO ===")
    print(best)
    
    # 5) Mostrar distribución de RMSE
    mostrar_distribucion_rmse(todos_los_rmse)

if __name__ == "__main__":
    main()
