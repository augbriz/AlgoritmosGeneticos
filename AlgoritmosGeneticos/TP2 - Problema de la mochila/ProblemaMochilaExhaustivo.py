
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

# Mostrar solo la mejor variante
print("Mejor combinación encontrada:")
print("Cromosoma:", Mejor_cromosoma)
print("Peso total:", mejor_peso)
print("Valor total:", mejor_valor)
print("Fitness:", mejor_fitness)
