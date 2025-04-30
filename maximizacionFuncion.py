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
    print("Cromosoma: ", gen_poblacion()[i])
    print("Decimal: ", cromosoma_decimal(poblacion[i]))
    print("Valor funcion: ", fitness(poblacion[i]))
    print("Fitness: ", (fitness(poblacion[i]))/fitness_total(poblacion))
    print("----------------------------------------------------")
    
