import numpy as np
import random


ciudades = ["La Plata","Córdoba","Corrientes","Paraná","Formosa","San Salvador de Jujuy","Santa Rosa","La Rioja","Mendoza","Posadas","Neuquén","Viedma","Salta",
"Rawson","San Juan","San Luis","Río Gallegos","Santa Fe","Santiago del Estero","Ushuaia","San Miguel de Tucumán","Resistencia","Catamarca"]
distancias = np.array([
    # LP   CBA  COR  PAR  FOR  JUJ  SRO  LRJ  MZA  POS  NEU  VIE  SAL  RAW  SJU  SLS  RGA  SFE  SDE  USH  TUC  RES  CAT
    [  0,  600, 800, 700, 900, 1600, 900, 1100, 1000, 1100, 1300, 1400, 1500, 1700, 1200, 1300, 2500, 500, 1000, 3000, 1200, 1100, 1100], # La Plata
    [600,    0, 200, 300, 700, 1100, 700, 400, 600, 900, 1100, 1200, 1300, 1500, 800, 900, 2200, 400, 700, 2700, 900, 800, 700], # Córdoba
    [800,  200,   0, 200, 500, 900, 900, 600, 800, 700, 1300, 1400, 1100, 1700, 1000, 1100, 2400, 600, 900, 2900, 1100, 600, 900], # Corrientes
    [700,  300, 200,   0, 400, 800, 800, 500, 700, 600, 1200, 1300, 1000, 1600, 900, 1000, 2300, 500, 800, 2800, 1000, 500, 800], # Paraná
    [900,  700, 500, 400,   0, 1000, 1100, 800, 1000, 300, 1500, 1600, 1200, 1900, 1200, 1300, 2600, 800, 1100, 3100, 1300, 400, 1100], # Formosa
    [1600, 1100, 900, 800, 1000,   0, 1500, 700, 900, 1200, 1700, 1800, 400, 2100, 1000, 1100, 2800, 1300, 600, 3300, 800, 1000, 700], # Jujuy
    [900,  700, 900, 800, 1100, 1500,   0, 900, 700, 1300, 400, 500, 1700, 600, 800, 700, 2000, 800, 1100, 2500, 900, 1200, 900], # Santa Rosa
    [1100, 400, 600, 500, 800, 700, 900,   0, 300, 1000, 900, 1000, 800, 1300, 200, 300, 1700, 700, 400, 2200, 500, 800, 200], # La Rioja
    [1000, 600, 800, 700, 1000, 900, 700, 300,   0, 1200, 700, 800, 1000, 1100, 400, 200, 1500, 900, 600, 2000, 700, 1000, 400], # Mendoza
    [1100, 900, 700, 600, 300, 1200, 1300, 1000, 1200,   0, 1700, 1800, 1400, 2100, 1400, 1500, 2600, 1100, 1400, 3100, 1500, 600, 1300], # Posadas
    [1300, 1100, 1300, 1200, 1500, 1700, 400, 900, 700, 1700,   0, 200, 1800, 300, 1100, 1000, 1600, 1000, 1300, 2100, 1200, 1700, 900], # Neuquén
    [1400, 1200, 1400, 1300, 1600, 1800, 500, 1000, 800, 1800, 200,   0, 1900, 100, 1200, 1100, 1700, 1100, 1400, 2200, 1300, 1800, 1000], # Viedma
    [1500, 1300, 1100, 1000, 1200, 400, 1700, 800, 1000, 1400, 1800, 1900,   0, 2200, 1200, 1300, 2900, 1400, 900, 3400, 1000, 1200, 900], # Salta
    [1700, 1500, 1700, 1600, 1900, 2100, 600, 1300, 1100, 2100, 300, 100, 2200,   0, 1400, 1300, 1900, 1300, 1600, 2400, 1500, 2100, 1200], # Rawson
    [1200, 800, 1000, 900, 1200, 1000, 800, 200, 400, 1400, 1100, 1200, 1200, 1400,   0, 100, 1800, 900, 600, 2300, 700, 1200, 300], # San Juan
    [1300, 900, 1100, 1000, 1300, 1100, 700, 300, 200, 1500, 1000, 1100, 1300, 1300, 100,   0, 1700, 1000, 700, 2200, 800, 1300, 400], # San Luis
    [2500, 2200, 2400, 2300, 2600, 2800, 2000, 1700, 1500, 2600, 1600, 1700, 2900, 1900, 1800, 1700,   0, 2100, 2400, 400, 1800, 2600, 1700], # Río Gallegos
    [500, 400, 600, 500, 800, 1300, 800, 700, 900, 1100, 1000, 1100, 1400, 1300, 900, 1000, 2100,   0, 300, 2600, 700, 800, 700], # Santa Fe
    [1000, 700, 900, 800, 1100, 600, 1100, 400, 600, 1400, 1300, 1400, 900, 1600, 600, 700, 2400, 300,   0, 2900, 400, 1100, 400], # Santiago del Estero
    [3000, 2700, 2900, 2800, 3100, 3300, 2500, 2200, 2000, 3100, 2100, 2200, 3400, 2400, 2300, 2200, 400, 2600, 2900,   0, 2300, 3100, 2200], # Ushuaia
    [1200, 900, 1100, 1000, 1300, 800, 900, 500, 700, 1500, 1200, 1300, 1000, 1500, 700, 800, 1800, 700, 400, 2300,   0, 1300, 500], # San Miguel de Tucumán
    [1100, 800, 600, 500, 400, 1000, 1200, 800, 1000, 600, 1700, 1800, 1200, 2100, 1200, 1300, 2600, 800, 1100, 3100, 1300,   0, 1100], # Resistencia
    [1100, 700, 900, 800, 1100, 700, 900, 200, 400, 1300, 900, 1000, 900, 1200, 300, 400, 1700, 700, 400, 2200, 500, 1100,   0], # Catamarca
])
n_ciudades = len(ciudades)
tam_poblacion = 500
while True:
    try:
        generaciones = int(input("Ingrese el número de generaciones: "))
        if generaciones > 0:
            break
        else:
            print("Por favor, ingrese un número entero positivo: ")
    except ValueError:
        print("Entrada inválida. Ingrese el número de generaciones: ")
tasa_mutacion = 0.1

def crear_individuo(n):
    ruta = list(range(n))
    random.shuffle(ruta)
    return ruta

def crear_poblacion(n_ciudades, tam_poblacion):
    return [crear_individuo(n_ciudades) for _ in range(tam_poblacion)]

def calcular_distancia(ruta, distancias):
    total = 0
    for i in range(len(ruta)):
        ciudad_origen = ruta[i]
        ciudad_destino = ruta[(i + 1) % len(ruta)]  # ciclo
        total += distancias[ciudad_origen][ciudad_destino]
    return total

def fitness(ruta, distancias):
    return 1 / calcular_distancia(ruta, distancias)

def crossover(padre1, padre2):
    start, end = sorted(random.sample(range(len(padre1)), 2))
    hijo = [None] * len(padre1)
    hijo[start:end+1] = padre1[start:end+1]

    fill = [c for c in padre2 if c not in hijo]
    i = 0
    for j in range(len(hijo)):
        if hijo[j] is None:
            hijo[j] = fill[i]
            i += 1
    return hijo  # <-- Corregido: ahora retorna el hijo

def mutar(ruta, tasa_mutacion):
    if random.random() < tasa_mutacion:
        i, j = random.sample(range(len(ruta)), 2)
        ruta[i], ruta[j] = ruta[j], ruta[i]
    return ruta

def seleccion_por_torneo(poblacion, distancias, k=3):
    seleccionados = random.sample(poblacion, k)
    seleccionados.sort(key=lambda ind: calcular_distancia(ind, distancias))
    return seleccionados[0]

def algoritmo_genetico():
    global distancias, tam_poblacion, tasa_mutacion, n_ciudades
    poblacion = crear_poblacion(n_ciudades, tam_poblacion)

    mejor = min(poblacion, key=lambda ind: calcular_distancia(ind, distancias))
    
    for gen in range(generaciones):
        nueva_poblacion = []
        for _ in range(tam_poblacion):
            padre1 = seleccion_por_torneo(poblacion, distancias)
            padre2 = seleccion_por_torneo(poblacion, distancias)
            hijo = crossover(padre1, padre2)
            hijo = mutar(hijo, tasa_mutacion)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion
        candidato = min(poblacion, key=lambda ind: calcular_distancia(ind, distancias))
        if calcular_distancia(candidato, distancias) < calcular_distancia(mejor, distancias):
            mejor = candidato

        if gen % 50 == 0:
            print(f"Generación {gen}: Mejor distancia = {calcular_distancia(mejor, distancias):.2f}")

    return mejor

mejor_ruta = algoritmo_genetico()
print([ciudades[i] for i in mejor_ruta])
print(calcular_distancia(mejor_ruta, distancias))