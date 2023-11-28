import random
import pandas as pd
import numpy as np

rutas = []  # lista de rutas
tamanio_poblacion = 10  # tamaño de la población
probabilidad_mutacion = 0.1
n_generaciones = 100  # número de generaciones
longitudes_rutas = [0] * tamanio_poblacion
aptitud = [0] * tamanio_poblacion
mejor_camino = 1000
mejor = []

ciudades = [0, 1, 2, 3, 4]
ciudades_x = [1, 2, 3, 4]

df = pd.read_csv(r'agente.csv', skiprows=0, low_memory=False)
distancias = np.array(df.values)

# Calcular distancia
def calcular_distancia(ciudad1, ciudad2):
    return distancias[ciudad1][ciudad2]

# Crear Ruta
def crear_ruta():
    aleatoria = [0]
    aleatoria += random.sample(ciudades_x, len(ciudades_x))
    return aleatoria

# Calcular la longitud de una ruta
def calcular_longitud_ruta():
    for i in range(tamanio_poblacion):
        longitud_ruta = 0
        for j in range(0, len(ciudades)):
            longitud_ruta = longitud_ruta + calcular_distancia(rutas[i][j - 1], rutas[i][j])
        longitudes_rutas[i] = longitud_ruta
        aptitud[i] = 1 / longitudes_rutas[i]

# Crear población
def crear_poblacion():
    for i in range(tamanio_poblacion):
        rutas.append(crear_ruta())

# Mutación
def mutacion_swap(ind):
    seleccionados = random.sample(range(len(ciudades)), 2)
    temp = rutas[ind][seleccionados[0]]
    rutas[ind][seleccionados[0]] = rutas[ind][seleccionados[1]]
    rutas[ind][seleccionados[1]] = temp

# Cruce
def cruza_parcialmente_coincidente(ind1, ind2):
    size = len(ciudades)
    p1, p2 = [0] * size, [0] * size

    # Inicializar la posición de cada índice en los individuos
    for k in range(size):
        p1[ind1[k]] = k
        p2[ind2[k]] = k
    # Elegir puntos de cruce
    punto_cruce1 = random.randint(0, size)
    punto_cruce2 = random.randint(0, size - 1)
    if punto_cruce2 >= punto_cruce1:
        punto_cruce2 += 1
    else:  # Intercambiar los dos puntos de cruce
        punto_cruce1, punto_cruce2 = punto_cruce2, punto_cruce1

    # Aplicar cruce
    for k in range(punto_cruce1, punto_cruce2):
        temp1 = ind1[k]
        temp2 = ind2[k]
        # Intercambio del valor coincidente
        ind1[k], ind1[p1[temp2]] = temp2, temp1
        ind2[k], ind2[p2[temp1]] = temp1, temp2
        # Actualización de las posiciones
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2

# Selección por ruleta
def seleccion_ruleta():
    s = 0
    suma_parcial = 0
    indice = 0
    for m in range(tamanio_poblacion):
        s = s + aptitud[m]
    aleatorio = random.uniform(0, s)
    for m in range(tamanio_poblacion):
        if suma_parcial < aleatorio:
            suma_parcial = suma_parcial + aptitud[m]
            indice = indice + 1
    # Evitar exceder el límite
    if indice == tamanio_poblacion:
        indice = tamanio_poblacion - 1
    return indice

# Encontrar el camino más corto
def encontrar_mejor():
    clave = 1000
    mejor_camino_indice = 0
    for i in range(tamanio_poblacion):
        if longitudes_rutas[i] < clave:  # Verifica y de ese modo el mejor es el mas pequeño (minimizacion)
            clave = longitudes_rutas[i] # Cambia por el menor
            mejor_camino_indice = i
    return mejor_camino_indice

crear_poblacion()
calcular_longitud_ruta()
mejor = [0, 0, 0, 0, 0]
rutas_generadas = []
recorrido = []
x = []

diccionario_rutas = {}

for j in range(n_generaciones):
    for i in range(0, tamanio_poblacion, 2):
        padre1 = seleccion_ruleta()
        padre2 = seleccion_ruleta()
        while True:
            if padre1 == padre2:
                padre2 = seleccion_ruleta()
            else:
                break
        rutas[i], rutas[i + 1] = cruza_parcialmente_coincidente(rutas[padre1], rutas[padre2])
        calcular_longitud_ruta()
    for i in range(tamanio_poblacion):
        aleatorio = random.uniform(0, 1)
        if aleatorio < probabilidad_mutacion:
            mutacion_swap(i)
    calcular_longitud_ruta()
    if longitudes_rutas[encontrar_mejor()] < mejor_camino:
        indice = encontrar_mejor()
        mejor_camino = longitudes_rutas[indice]
    x = rutas[encontrar_mejor()]
    print("Camino ", j + 1, ": ", x, "Recorrido: ", longitudes_rutas[encontrar_mejor()])
    mejor += rutas[encontrar_mejor()]
    rutas_generadas.append(longitudes_rutas[encontrar_mejor()])
    if longitudes_rutas[encontrar_mejor()] == mejor_camino:
        gr = j
    recorrido.append(x)

ind = 5 * (gr + 1) # El indice de la mejor ruta
print("\nMejor Camino: [", mejor[ind], mejor[ind + 1], mejor[ind + 2], mejor[ind + 3], mejor[ind + 4], "]\nDistancia:", mejor_camino)
df_rutas = pd.DataFrame({'Camino': recorrido, 'Recorridos': rutas_generadas})
df_rutas.to_csv('rutas.csv', index=True)