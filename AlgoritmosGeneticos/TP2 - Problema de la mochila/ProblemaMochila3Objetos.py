import time
import sys
import os
os.system('chcp 65001')  # Cambiar codificación a UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Datos de los objetos: [posición, peso en gramos, valor en $]
objetos = [
    [1, 1800, 72],
    [2, 600, 36],
    [3, 1200, 60]
]

# Capacidad máxima de la mochila en gramos
capacidad = 3000

def busqueda_exhaustiva(objetos, capacidad):
    n = len(objetos)
    mejor_fitness = -1
    mejor_peso = None
    mejor_valor = None
    mejor_cromosoma = None

    tiempo_inicio = time.time()
    
    # Recorrer todas las combinaciones posibles (2^3 = 8 combinaciones)
    for mask in range(1 << n):
        peso = 0
        valor = 0
        cromosoma = []
        for i in range(n):
            bit = (mask >> i) & 1
            cromosoma.append(bit)
            if bit:
                peso += objetos[i][1]
                valor += objetos[i][2]
        
        # Calcular fitness (0 si excede la capacidad)
        fitness = valor if peso <= capacidad else 0
        
        # Guardar si es la mejor combinación encontrada
        if fitness > mejor_fitness or (fitness == mejor_fitness and (mejor_peso is None or peso < mejor_peso)):
            mejor_fitness = fitness
            mejor_peso = peso
            mejor_valor = valor
            mejor_cromosoma = cromosoma[:]

    tiempo_fin = time.time()
    return mejor_cromosoma, mejor_peso, mejor_valor, tiempo_fin - tiempo_inicio

def solucion_greedy(objetos, capacidad):
    tiempo_inicio = time.time()
    
    # Calcular la relación valor/peso para cada objeto
    objetos_valor_peso = [(obj[0], obj[1], obj[2], obj[2]/obj[1]) for obj in objetos]
    
    # Ordenar objetos por relación valor/peso de mayor a menor
    objetos_ordenados = sorted(objetos_valor_peso, key=lambda x: x[3], reverse=True)
    
    peso_actual = 0
    valor_total = 0
    cromosoma = [0] * len(objetos)
    
    # Seleccionar objetos mientras quede capacidad
    for obj in objetos_ordenados:
        if peso_actual + obj[1] <= capacidad:
            peso_actual += obj[1]
            valor_total += obj[2]
            cromosoma[obj[0]-1] = 1
    
    tiempo_fin = time.time()
    return cromosoma, peso_actual, valor_total, tiempo_fin - tiempo_inicio

# Ejecutar búsqueda exhaustiva
print("\n=== SOLUCIÓN CON BÚSQUEDA EXHAUSTIVA ===")
cromosoma_exh, peso_exh, valor_exh, tiempo_exh = busqueda_exhaustiva(objetos, capacidad)
print(f"Cromosoma: {cromosoma_exh}")
print(f"Peso total: {peso_exh} gramos")
print(f"Valor total: ${valor_exh}")
print(f"Tiempo de ejecución: {tiempo_exh:.6f} segundos")

# Ejecutar algoritmo greedy
print("\n=== SOLUCIÓN CON ALGORITMO GREEDY ===")
cromosoma_greedy, peso_greedy, valor_greedy, tiempo_greedy = solucion_greedy(objetos, capacidad)
print(f"Cromosoma: {cromosoma_greedy}")
print(f"Peso total: {peso_greedy} gramos")
print(f"Valor total: ${valor_greedy}")
print(f"Tiempo de ejecución: {tiempo_greedy:.6f} segundos")

# Comparar resultados
print("\n=== COMPARACIÓN DE RESULTADOS ===")
diferencia_valor = valor_exh - valor_greedy
print(f"Diferencia en valor: ${diferencia_valor}")

# Análisis de la solución
print("\n=== ANÁLISIS DE LA SOLUCIÓN ===")
print("Relaciones valor/peso de cada objeto:")
for obj in objetos:
    print(f"Objeto {obj[0]}: ${obj[2]}/{obj[1]}g = {obj[2]/obj[1]:.4f} $/g")

if diferencia_valor == 0:
    print("\nAmbos algoritmos encontraron una solución óptima!")
elif diferencia_valor > 0:
    print(f"\nLa búsqueda exhaustiva encontró una mejor solución por ${diferencia_valor}")
else:
    print(f"\nEl algoritmo greedy encontró una mejor solución por ${-diferencia_valor}")