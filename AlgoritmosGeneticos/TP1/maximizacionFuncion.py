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
ciclos = int(input("cantidad de ciclos a realizar: "))
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

# Selección por torneo: toma un porcentaje de la población y elige el de mayor fitness
def seleccion_torneo(poblacion, porcentaje=0.4):
    global pob_ini
    # nueva_pobl almacenará los cromosomas ganadores de cada torneo
    nueva_pobl = []

    tam_torneo = (int(len(poblacion) * porcentaje))
    # repetimos hasta seleccionar n_select cromosomas
    for _ in range(pob_ini):
        # elegimos al azar 'tam_torneo' participantes del torneo
        participantes = random.sample(poblacion, tam_torneo)
        # de esos participantes, seleccionamos el de mayor fitness
        ganador = max(participantes, key=fitness)
        # agregamos el ganador a la nueva lista
        nueva_pobl.append(ganador)
    # devolvemos la población resultante tras todos los torneos
    return nueva_pobl

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

# 2) Selección de método
print("Seleccione el método de selección:\n1) Ruleta\n2) Torneo")
sel_method = input("Ingrese 1 o 2: ")

# 3) Si eligió ruleta, preguntar por elitismo
elitismo = False
if sel_method == '1':
    print("¿Desea implementar elitismo? (s/n)")
    elitismo = (input().lower() == 's')

# Lista para guardar el fitness promedio en cada ciclo, Aqui se debería guardar tambien el promedio de la funcion objetivo, maximo y minimo de cada población
avg_fitness = []

# Listas para guardar los valores promedio, máximo y mínimo de la función objetivo
prom_f_obj = []
max_f_obj  = []
min_f_obj  = []
# Lista para guardar el mejor cromosoma de cada ciclo
mejor_cromosoma = []

for i in range (pob_ini):
    print("Numero de cromosoma: ", i+1)
    print("Cromosoma: ", poblacion[i])
    print("Decimal: ", cromosoma_decimal(poblacion[i]))
    print("Valor funcion: ", fitness(poblacion[i]))
    print("Fitness: ", (fitness(poblacion[i]))/fitness_total(poblacion))
    print("----------------------------------------------------")
    

for i in range(ciclos):
    print(Fore.GREEN + f"\n---------------------------------------------------- Ciclo: {i+1} ----------------------------------------------------\n" + Style.RESET_ALL)
    total_fit = fitness_total(poblacion)
    
    # 2) Preparamos las columnas y calculamos fitness de cada cromosoma
    nums      = list(range(1, pob_ini+1))
    crom_s    = [''.join(str(b) for b in c) for c in poblacion]
    decimales = [cromosoma_decimal(c) for c in poblacion]
    f_objs    = [fitness(c) for c in poblacion]
    fits      = [f/total_fit for f in f_objs]
    
    # Calculamos y almacenamos los valores promedio, máximo y mínimo de la función objetivo
    promedio = sum(f_objs) / len(poblacion)
    maximo  = max(f_objs)
    minimo  = min(f_objs)
    
    prom_f_obj.append(promedio)
    max_f_obj.append(maximo)
    min_f_obj.append(minimo)
    # almacenar cromosoma con fitness máximo
    mejor_cromosoma.append(crom_s[f_objs.index(maximo)])
    
    # 3) DataFrame principal
    df = pd.DataFrame({
        'Num':        nums,
        'Cromosoma':  crom_s,
        'Decimal':    decimales,
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
    
    # Generación de nueva población
    if sel_method == '1':
        if elitismo == 's':
            # extraemos dos mejores directos (sin crossover ni mutación)
            elite = sorted(poblacion, key=fitness, reverse=True)[:2]
            # resto de la población vía ruleta
            resto = [seleccion_ruleta(poblacion) for _ in range(pob_ini - 2)]
            # aplicamos crossover y mutación solo al resto
            resto = crossover_un_corte(resto, prob_cross)
            resto = mutacion(resto, prob_mut)
            nueva_poblacion = elite + resto
        else:
            # ruleta completa con crossover y mutación
            tmp = [seleccion_ruleta(poblacion) for _ in range(pob_ini)]
            nueva_poblacion = crossover_un_corte(tmp, prob_cross)
            nueva_poblacion = mutacion(nueva_poblacion, prob_mut)
    elif sel_method == '2':
        # torneo completo con crossover y mutación
        tmp = seleccion_torneo(poblacion, 0.4)
        nueva_poblacion = crossover_un_corte(tmp, prob_cross)
        nueva_poblacion = mutacion(nueva_poblacion, prob_mut)
    else:
        print("Método no válido, usando ruleta por defecto")
        tmp = [seleccion_ruleta(poblacion) for _ in range(pob_ini)]
        nueva_poblacion = crossover_un_corte(tmp, prob_cross)
        nueva_poblacion = mutacion(nueva_poblacion, prob_mut)
    
    poblacion = nueva_poblacion  # Actualizamos la población para la siguiente iteración
     

# Después de completar los ciclos, graficamos:
plt.figure()
plt.plot(range(1, ciclos+1), prom_f_obj, marker='o', label='Promedio F_obj')
plt.plot(range(1, ciclos+1), max_f_obj, marker='x', label='Máximo F_obj', color='red')
plt.plot(range(1, ciclos+1), min_f_obj, marker='s', label='Mínimo F_obj', color='green')
plt.title('Evolución de la función objetivo')
plt.xlabel('Ciclo')
plt.ylabel('Valor de la función objetivo')
plt.legend()
plt.grid(True)

# Tabla final de resumen de cada ciclo
df_final = pd.DataFrame({
    'Ciclo':           list(range(1, ciclos+1)),
    'Máximo F_obj':   max_f_obj,
    'Mínimo F_obj':   min_f_obj,
    'Promedio F_obj': prom_f_obj,
    'Mejor Cromosoma': mejor_cromosoma
})
print("\nTabla Resumen:")
print("Cantidad de ciclos:", ciclos)
print(df_final.to_string(index=False))

# mostramos el gráfico 
plt.show()


