import pandas as pd
import random 
import heapq
#Fijamos el valor de las variables a utilizar
prob_cross = 0.75
prob_mut = 0.05
pob_ini = 10 
ciclos = 20
gen_size = 30
coef = 2**30 - 1
poblacion = []

# Funcion(x/coef)^2 
    
    
#Definimos los cromosomas iniciales 
def gen_poblacion(): 
    for i in range(pob_ini):
        cromosoma = []
        for j in range(gen_size):
         cromosoma.append(random.randint(0, 1))
        poblacion.append(cromosoma)
        
    return poblacion
#Generamos la poblacion inicial 
poblacion = gen_poblacion()
#Cromosoma a decimial
def cromosoma_decimal(cromosoma):
    decimal = 0
    for i in range(gen_size):
        decimal += cromosoma[i] * (2 ** i)
    return decimal


#Valuamos la funcion por cromosoma
def fitness(cromosoma):
 decimal = cromosoma_decimal(cromosoma)
 return (decimal/coef)**2


#Sacamos porcentaje de fitness

def fitness_total(poblacion):
    suma=0
    for i in range (pob_ini):
        suma+= fitness(poblacion[i])
    return suma    


for i in range (pob_ini):
    print("Numero de cromosoma: ", i+1)
    print("Cromosoma: ", poblacion[i])
    print("Decimal: ", cromosoma_decimal(poblacion[i]))
    print("Valor funcion: ", fitness(poblacion[i]))
    print("Fitness: ", (fitness(poblacion[i]))/fitness_total(poblacion))
    print("----------------------------------------------------")
    

total_fit = fitness_total(poblacion)
# 2) Preparamos las columnas
nums      = list(range(1, pob_ini+1))
crom_s    = [''.join(str(b) for b in c) for c in poblacion]
decimals  = [cromosoma_decimal(c) for c in poblacion]
f_objs    = [fitness(c) for c in poblacion]
fits      = [f/total_fit for f in f_objs]

# 3) DataFrame principal
df = pd.DataFrame({
    'Num':        nums,
    'Cromosoma':  crom_s,
    'Decimal':    decimals,
    'F_obj':      f_objs,
    'Fitness':    fits
})

# 4) Fila de resumen: suma, promedio y máximo
resumen = pd.DataFrame({
    'Num':        ['suma', 'promedio', 'maximo'],
    'Cromosoma':  ['', '', ''],
    'Decimal':    [df['Decimal'].sum(), df['Decimal'].mean(), df['Decimal'].max()],
    'F_obj':      [df['F_obj'].sum(),    df['F_obj'].mean(),    df['F_obj'].max()],
    'Fitness':    [df['Fitness'].sum(),  df['Fitness'].mean(),  df['Fitness'].max()]
})

df_res = pd.concat([df, resumen], ignore_index=True)

# 5) Imprimimos la tabla formateada
print(df_res.to_string(index=False, 
    formatters={
      'Decimal': '{:.0f}'.format,
      'F_obj':   '{:.2f}'.format,
      'Fitness': '{:.2f}'.format
    }
))

#Seleccion por ruleta

def seleccion_ruleta(poblacion):
    # Calculamos la suma total de fitness
    total_fit = fitness_total(poblacion)
    
    # Creamos un array de 100 posiciones
    ruleta = []
    
    # Asignamos cromosomas al array según su proporción de fitness
    for i, cromosoma in enumerate(poblacion):
        # Calculamos cuántas posiciones ocupa este cromosoma en la ruleta
        proporciones = int((fitness(cromosoma) / total_fit) * 100)
        ruleta.extend([i] * proporciones)  # Añadimos el índice del cromosoma
    
    # Sorteamos una posición en la ruleta
    posicion = random.randint(0, len(ruleta) - 1)
    
    # Devolvemos el cromosoma seleccionado
    return poblacion[ruleta[posicion]]

nueva_poblacion_ruleta = []  # Inicializamos la nueva población como una lista vacía
for _ in range(10):  # Iteramos 10 veces para generar 10 cromosomas
    nueva_poblacion_ruleta .append(seleccion_ruleta(poblacion))  # Agregamos el cromosoma seleccionado a la nueva población

# Imprimimos la nueva población generada
print("Nueva población generada:")
for i, cromosoma in enumerate(nueva_poblacion_ruleta , 1):
    print(f"Cromosoma {i}: {cromosoma}")
    
#Realizamos crossover
# Realizamos el crossover de un corte con probabilidad
def crossover_un_corte(nueva_poblacion_ruleta, prob_cross):
    nueva_poblacion_crossover = []
    for i in range(0, len(nueva_poblacion_ruleta), 2):  # Iteramos en pares
        padre1 = nueva_poblacion_ruleta[i]
        padre2 = nueva_poblacion_ruleta[i + 1]
        
        # Decidimos si realizar el crossover según la probabilidad
        if random.random() < prob_cross:
            # Elegimos un punto de corte al azar
            punto_corte = random.randint(1, gen_size - 1)  # Entre 1 y gen_size-1
            
            # Creamos los hijos intercambiando genes en el punto de corte
            hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
            hijo2 = padre2[:punto_corte] + padre1[punto_corte:]
        else:
            # Si no se realiza crossover, los hijos son copias de los padres
            hijo1 = padre1
            hijo2 = padre2
        
        # Añadimos los hijos a la nueva población
        nueva_poblacion_crossover.append(hijo1)
        nueva_poblacion_crossover.append(hijo2)
    
    return nueva_poblacion_crossover

# Aplicamos el crossover a la nueva población con probabilidad prob_cross
Nueva_poblacion_crossover = crossover_un_corte(nueva_poblacion_ruleta, prob_cross)

# Imprimimos la nueva población después del crossover
print("Nueva población después del crossover:")
for i, cromosoma in enumerate(Nueva_poblacion_crossover, 1):
    print(f"Cromosoma {i}: {cromosoma}")
    
#Realizamos mutacion
def mutacion(Nueva_poblacion_crossover, prob_mut):
    for j in range(len(Nueva_poblacion_crossover)):
        for i in range(len(Nueva_poblacion_crossover[j])):
            if random.random() < prob_mut:
                # Cambiamos el bit
                print ("Habemus mutacion")
                Nueva_poblacion_crossover[j][i] = 1 - Nueva_poblacion_crossover[j][i]  # Cambia de 0 a 1 o de 1 a 0
        return Nueva_poblacion_crossover
    
# Imprimimos la nueva población después de la mutación
Nueva_poblacion_mutacion = mutacion(Nueva_poblacion_crossover, prob_mut)
print("Nueva población después del mutación:")
for i, cromosoma in enumerate(Nueva_poblacion_mutacion, 1):
    print(f"Cromosoma {i}: {cromosoma}")