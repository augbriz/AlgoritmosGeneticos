import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Descargar datos
ticker = "YPF"
data = yf.download(ticker, start="2025-01-01", end="2025-07-23")
serie = data['Close'].dropna().values

# Parámetros del GA
WINDOW_SIZE = 5        # cuántos valores atrás usar
POP_SIZE = 50          # tamaño de la población
GENERATIONS = 100      # número de generaciones
MUTATION_RATE = 0.1    # probabilidad de mutación
RANGE = (-2, 2)        # rango de pesos aleatorios

# Crear dataset (ventanas de precios)
X = []
y = []
for i in range(WINDOW_SIZE, len(serie)):
    X.append(serie[i-WINDOW_SIZE:i])
    y.append(serie[i])
X = np.array(X).reshape(-1, WINDOW_SIZE)  # <- esta línea es clave
y = np.array(y)

def init_population(pop_size, n_features):
    return np.random.uniform(RANGE[0], RANGE[1], size=(pop_size, n_features))

def fitness(individuo, X, y):
    preds = np.dot(X, individuo.flatten())

    return -np.mean((preds - y) ** 2)

def select(population, scores):
    idx1, idx2 = np.random.choice(len(population), 2, replace=False)
    return population[idx1] if scores[idx1] > scores[idx2] else population[idx2]

def crossover(parent1, parent2):
    return (parent1 + parent2) / 2

def mutate(individuo, rate):
    for i in range(len(individuo)):
        if np.random.rand() < rate:
            individuo[i] += np.random.normal(0, 0.1)  # mutación gaussiana
    return np.array(individuo).flatten()


population = init_population(POP_SIZE, WINDOW_SIZE)

best_fitness = -np.inf
best_individual = None
history = []

for generation in range(GENERATIONS):
    scores = np.array([fitness(ind, X, y) for ind in population])
    
    # Guardar el mejor
    max_idx = np.argmax(scores)
    if scores[max_idx] > best_fitness:
        best_fitness = scores[max_idx]
        best_individual = population[max_idx]
    
    history.append(-best_fitness)  # error positivo
    
    # Crear nueva población
    new_population = []
    for _ in range(POP_SIZE):
        p1 = select(population, scores)
        p2 = select(population, scores)
        child = crossover(p1, p2)
        child = mutate(child, MUTATION_RATE)
        new_population.append(child)
    
    population = np.array(new_population)

print("Mejor individuo (pesos):", best_individual)
print("Error cuadrático medio:", -best_fitness)

plt.plot(history)
plt.title("Evolución del error")
plt.xlabel("Generación")
plt.ylabel("MSE")
plt.grid()
plt.show()

# Última ventana disponible
ultimos_valores = serie[-WINDOW_SIZE:]
prediccion = np.dot(best_individual, ultimos_valores)
print("Predicción del próximo precio:", prediccion)
