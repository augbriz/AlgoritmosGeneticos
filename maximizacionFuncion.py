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
#Generamos la poblacion inicial #CAMBIO 
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

