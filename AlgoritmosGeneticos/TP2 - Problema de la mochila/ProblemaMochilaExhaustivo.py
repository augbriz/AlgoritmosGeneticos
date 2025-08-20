import sys
import os
os.system('chcp 65001')  # Cambiar codificación a UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Datos de los objetos: [posición, peso, valor]
objetos = [
    [1, 150, 20], [2, 325, 40], [3, 600, 50], [4, 805, 36], [5, 430, 25],
    [6, 1200, 64], [7, 770, 54], [8, 60, 18], [9, 930, 46], [10, 353, 28]
]

# Capacidad máxima de la mochila
capacidad = 4200

n = len(objetos)
mejor_fitness = -1
mejor_peso = None
mejor_valor = None
Mejor_cromosoma = None  # Variable para guardar el mejor cromosoma

# Recorrido exhaustivo de todas las combinaciones
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
        Mejor_cromosoma = cromosoma[:]



def solucion_greedy(objetos, capacidad):
    # Calcular la relación valor/peso para cada objeto
    objetos_valor_peso = [(obj[0], obj[1], obj[2], obj[2]/obj[1]) for obj in objetos]
    
    # Ordenar objetos por relación valor/peso de mayor a menor
    objetos_ordenados = sorted(objetos_valor_peso, key=lambda x: x[3], reverse=True)
    
    peso_actual = 0
    valor_total = 0
    seleccionados = []
    cromosoma = [0] * len(objetos)
    
    # Seleccionar objetos mientras quede capacidad
    for obj in objetos_ordenados:
        if peso_actual + obj[1] <= capacidad:
            peso_actual += obj[1]
            valor_total += obj[2]
            seleccionados.append(obj[0])
            cromosoma[obj[0]-1] = 1
    
    return cromosoma, peso_actual, valor_total

# Ejecutar búsqueda exhaustiva
print("\n=== SOLUCIÓN CON BÚSQUEDA EXHAUSTIVA ===")
print("Cromosoma:", Mejor_cromosoma)
print("Peso total:", mejor_peso)
print("Valor total:", mejor_valor)

# Ejecutar algoritmo greedy
print("\n=== SOLUCIÓN CON ALGORITMO GREEDY ===")
cromosoma_greedy, peso_greedy, valor_greedy = solucion_greedy(objetos, capacidad)
print("Cromosoma:", cromosoma_greedy)
print("Peso total:", peso_greedy)
print("Valor total:", valor_greedy)

# Comparar resultados
print("\n=== COMPARACIÓN DE RESULTADOS ===")
diferencia_valor = mejor_valor - valor_greedy
print(f"Diferencia en valor: {diferencia_valor}")
if diferencia_valor == 0:
    print("¡Ambos algoritmos encontraron una solución óptima!")
elif diferencia_valor > 0:
    print(f"La búsqueda exhaustiva encontró una mejor solución por {diferencia_valor} unidades")
else:
    print(f"El algoritmo greedy encontró una mejor solución por {-diferencia_valor} unidades")

