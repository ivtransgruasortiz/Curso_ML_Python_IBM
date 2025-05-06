import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler


def generar_datos_clientes(num_muestras):
    monto_total_gastado = np.random.uniform(100, 10000, size=num_muestras)
    frecuencia_compras = np.random.randint(1, 101, size=num_muestras)
    category = np.random.randint(0, 20, (num_muestras, 3))
    return np.array([monto_total_gastado, frecuencia_compras]), np.array(category)

def entrenar_modelo_cluster(datos):
    df1 = pd.DataFrame(datos[0], columns=['monto_total_gastado', 'frecuencia_compras'])
    df2 = pd.DataFrame(datos[1], columns=['categoria1', 'categoria2', 'categoria3'])
    datos = pd.concat([df1, df2], axis=1)
    kmeans = KMeans(3)
    scaler = StandardScaler()
    datos_sc = scaler.fit_transform(datos)
    modelo = kmeans.fit(datos_sc)
    return modelo, scaler


# def plot_results(inertials):
#     # x, y = zip(*[inertia for inertia in inertials])
#     x = [inertia[0] for inertia in inertials]
#     y = [inertia[1] for inertia in inertials]
#     plt.plot(x, y, 'ro-', markersize=8, lw=2)
#     plt.grid(True)
#     plt.xlabel('Num Clusters')
#     plt.ylabel('Inertia')
#     plt.show()

def encontrar_numero_optimo_clusters(datos, max_clusters=10):
    inertia_clusters = []
    for i in range(1, max_clusters):
        kmeans = entrenar_modelo_cluster(datos, i)
        inertia_clusters.append((i, kmeans.inertia_))
    plot_results(inertia_clusters)

def predecir_cluster(modelo, datos):
    df1 = pd.DataFrame(datos[0], columns=['monto_total_gastado', 'frecuencia_compras'])
    df2 = pd.DataFrame(datos[1], columns=['categoria1', 'categoria2', 'categoria3'])
    datosdf = pd.concat([df1, df2], axis=1)
    cliente1 = scaler.fit_transform(datosdf)
    prediction = modelo.predict(cliente1)[0]
    print("El cliente pertenece al cluster:", prediction)
    return prediction

datos = generar_datos_clientes(100)
modelo, scaler = entrenar_modelo_cluster(datos)

prueba = generar_datos_clientes(1)
predecir_cluster(modelo, prueba)

"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def generar_datos_clientes(num_muestras):
    np.random.seed(42)
    monto = np.random.uniform(100, 10000, num_muestras)
    frecuencia = np.random.randint(1, 101, num_muestras)
    categorias = np.random.randint(0, 20, (num_muestras, 3))
    total_categorias = categorias.sum(axis=1)
    datos = np.column_stack((monto, frecuencia, total_categorias))
    etiquetas = np.zeros(num_muestras)  # dummy necesario para test
    return datos, etiquetas
    
def entrenar_modelo_cluster(datos, n_clusters=3):
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos)
    modelo = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=42)
    modelo.fit(datos_escalados)
    return modelo, scaler
    
def predecir_cluster(modelo, scaler, cliente):
    cliente_array = np.array(cliente).reshape(1, -1)
    cliente_escalado = scaler.transform(cliente_array)
    return int(modelo.predict(cliente_escalado)[0])
    
def encontrar_numero_optimo_clusters(datos):
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos)

    mejor_k = 2
    mejor_score = -1
    # máximo 4 iteraciones para no explotar el runtime
    for k in [2, 3, 4, 5]:
        modelo = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
        etiquetas = modelo.fit_predict(datos_escalados)
        score = silhouette_score(datos_escalados, etiquetas)
        if score > mejor_score:
            mejor_score = score
            mejor_k = k

    return mejor_k
    

"""