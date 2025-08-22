# Resumen técnico (alto nivel)

Serie: Cierres diarios de YPF (últimos 1000 puntos), descargados con yfinance.

Métrica única para comparar corridasMaxScaler ajustado solo con el tramo de entrenamiento (para evitar fuga de datos).

Ventana temporal (lookback): 20 días de historia → entradas.

Horizonte de predicción (H): 5 días por delante → salidas multi-step.

Split temporal: 70% train, 15% validación, 15% test (sin mezclar, manteniendo orden).

Modelo: LSTM (1–3 capas), Dense de salida con H=5; loss MSE, métrica RMSE.

Callbacks: EarlyStopping (paciencia en val_loss) + ReduceLROnPlateau.

Fitness del GA: RMSE promedio normalizado en validación (promedio sobre los 5 horizontes).

También registramos el RMSE promedio en USD (interpretabilidad).

# Algoritmo Genético (qué hace y cómo lo implementamos)

El AG explora combinaciones de hiperparámetros de la LSTM. Cada individuo es un conjunto de hiperparámetros. Su aptitud (fitness) es el RMSE promedio normalizado en validación del modelo entrenado con esos hiperparámetros (menor es mejor).

## Representación del individuo (hiperparámetros)

Incluimos los siguientes genes en la población:

**lstm_units** (entero, p.ej. 32–96): tamaño de la capa LSTM (número de neuronas). Más unidades → más capacidad para modelar patrones, pero mayor riesgo de sobreajuste y costo computacional.

**num_layers** (entero, 1–3): profundidad (cuántas capas LSTM apiladas). Más capas pueden capturar patrones más complejos, pero también complican el entrenamiento.

**dropout_rate** (float, 0.05–0.35): regularización, desactiva aleatoriamente una fracción de neuronas durante entrenamiento para reducir sobreajuste.

**learning_rate** (float, ~1e-4–3e-3): tasa de aprendizaje del optimizador. La muestreamos y recombinamos en log-escala para tener saltos más estables (los órdenes de magnitud importan).

**batch_size** (categórico: {16, 32, 64}): tamaño de lote para fit. Afecta estabilidad y tiempo por época.

**epochs** (entero, 40–90): máximo de épocas. El entrenamiento puede detenerse antes por EarlyStopping; igualmente registramos best_epoch.

Fijos por ahora (fuera de la búsqueda): lookback_window = 20 y prediction_horizon = 5.
Si más adelante conviene, podemos incluir lookback_window en el espacio de búsqueda y/o probar horizontes alternativos.

## Inicialización

Población inicial (G0) muestreada al azar dentro de los rangos:

Uniforme para enteros y floats (con learning_rate en log-uniforme).

Aleatorio uniforme sobre las categorías para batch_size.

## Evaluación (función fitness)

Para cada individuo:

Construimos y entrenamos la LSTM con esos hiperparámetros.

Calculamos el RMSE por horizonte en validación (desnormalizado y normalizado).

El fitness es el RMSE promedio normalizado (promedio sobre t+1…t+5).

## Selección

Torneo de tamaño k=3: elegimos al azar 3 individuos y pasa el de mejor fitness.
Sencillo, robusto y sin requerir ordenamiento global en cada cruce.

## Cruce (crossover)

Enteros y categóricos: intercambio simple (50/50).

Floats:

dropout_rate: promedio de los padres + ruido gaussiano pequeño; acotado al rango.

learning_rate: mezcla en log-escala (promedio de log10(lr) de los padres) + ruido; acotado.

El acotado de valores se realiza dentro del operador (no usamos un clip global externo).

## Mutación

Probabilidad p por gen. Cambios pequeños y acotados:

lstm_units: ±4/±8

num_layers: ±1

epochs: ±5/±10

dropout_rate: ruido gaussiano pequeño

learning_rate: multiplicación por 10^N(0, σ) (log-perturbación)

batch_size: mover al vecino (e.g., de 32 a 16 o 64)

## Elitismo

Los 2 mejores de cada generación pasan directamente a la siguiente (garantiza no perder lo mejor ya encontrado).

## Bucle evolutivo

G0: evaluar población aleatoria y rankear.

Para cada generación G1..GN:

Copiar élites.

Repetir: selección por torneo → cruce → mutación, hasta completar la población.

Evaluar nueva población y rankear.

Reportar Mejor Global al final (fitness y RMSE_USD promedio).

## Hiperparámetros del AG usados (por defecto)

POP_SIZE = 8

N_GEN = 4 (ajustable)

ELITE_K = 2

TOURN_K = 3

CX_RATE = 0.9

MUT_RATE = 0.3

Estos valores son conservadores para correr en una laptop. Para una búsqueda “seria”, aumentar POP_SIZE y N_GEN, o ejecutar en varios seeds y promediar.

## Métrica para comparar corridas

RMSE promedio en USD (validación), que promedia los cinco horizontes t+1..t+5 desnormalizados.

Es fácil de entender para cualquier lector y refleja directamente la calidad del pronóstico en unidades reales. Internamente el AG optimiza la versión normalizada por estabilidad, pero siempre mostramos ambas.