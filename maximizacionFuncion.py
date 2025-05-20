import pandas as pd
import random 
import heapq
import matplotlib.pyplot as plt     # para graficar
from colorama import init, Fore, Style
init()

#Fijamos el valor de las variables a utilizar
prob_cross = 0.75
prob_mut = 0.05
pob_ini = 10 
ciclos = 50
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

#Realizamos mutacion
def mutacion(Nueva_poblacion_crossover, prob_mut):
    for j in range(len(Nueva_poblacion_crossover)):
        if random.random() < prob_mut:
            # Elegimos un punto de mutación al azar
            punto_mutacion = random.randint(0, gen_size - 1)
            print(Fore.RED + "\n----------------------------------------------------Habemus mutacion----------------------------------------------------" + Style.RESET_ALL)
            Nueva_poblacion_crossover[j][punto_mutacion] = 1 - Nueva_poblacion_crossover[j][punto_mutacion]  # Cambia de 0 a 1 o de 1 a 0
    return Nueva_poblacion_crossover
    
#Generamos la poblacion inicial 
poblacion = gen_poblacion()

# Lista para guardar el fitness promedio en cada ciclo
avg_fitness = []

for i in range (pob_ini):
    print("Numero de cromosoma: ", i+1)
    print("Cromosoma: ", poblacion[i])
    print("Decimal: ", cromosoma_decimal(poblacion[i]))
    print("Valor funcion: ", fitness(poblacion[i]))
    print("Fitness: ", (fitness(poblacion[i]))/fitness_total(poblacion))
    print("----------------------------------------------------")
    

for i in range(ciclos):
    print(Fore.GREEN +"'\n'""----------------------------------------------------Ciclo: ", i+1, "----------------------------------------------------", '\n' + Style.RESET_ALL)
    total_fit = fitness_total(poblacion)
    
    # 2) Preparamos las columnas y calculamos fitness de cada cromosoma
    nums      = list(range(1, pob_ini+1))
    crom_s    = [''.join(str(b) for b in c) for c in poblacion]
    decimals  = [cromosoma_decimal(c) for c in poblacion]
    f_objs    = [fitness(c) for c in poblacion]
    fits      = [f/total_fit for f in f_objs]
    
    # Calculamos y almacenamos el fitness promedio
    promedio = sum(f_objs) / len(poblacion)
    avg_fitness.append(promedio)
    
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
    print(df_res.to_string(index=False,
        formatters={
            'Decimal': '{:.0f}'.format,
            'F_obj':   '{:.2f}'.format,
            'Fitness': '{:.2f}'.format
        }
    ))
    
    # Generación de nueva población por ruleta, crossover y mutación...
    nueva_poblacion = []  # Inicializamos la nueva población
    for _ in range(10):  
        nueva_poblacion.append(seleccion_ruleta(poblacion))
    
    #print("'\n'----------------------------------------------------Nueva población generada: ----------------------------------------------------")
    #for i, cromosoma in enumerate(nueva_poblacion , 1):
        #print(f"Cromosoma {i}: {cromosoma}")
    
    nueva_poblacion = crossover_un_corte(nueva_poblacion, prob_cross)
    
    #print("'\n'----------------------------------------------------Nueva población después del crossover: ----------------------------------------------------")
    #for i, cromosoma in enumerate(nueva_poblacion, 1):
        #print(f"Cromosoma {i}: {cromosoma}")
        
    nueva_poblacion = mutacion(nueva_poblacion, prob_mut)
    
    #print("'\n'----------------------------------------------------Nueva población después del mutación: ----------------------------------------------------")
    #for i, cromosoma in enumerate(nueva_poblacion, 1):
        #print(f"Cromosoma {i}: {cromosoma}")
    
    poblacion = nueva_poblacion  # Actualizamos la población para la siguiente iteración

# Después de completar los ciclos, graficamos:
plt.figure()
plt.plot(range(1, ciclos+1), avg_fitness, marker='o')
plt.title('Evolución del fitness promedio')
plt.xlabel('Ciclo')
plt.ylabel('Fitness promedio')
plt.grid(True)
plt.show()



