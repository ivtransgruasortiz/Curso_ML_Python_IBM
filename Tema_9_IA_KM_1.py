import numpy as np
from sklearn.datasets import load_iris

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

def entrenar_y_evaluar_kmeans(X, y, k):
    # Entrenar el modelo K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    # Obtener el silhouette score
    silhouette = silhouette_score(X, kmeans.labels_)

    # Obtener el adjusted rand score
    adjusted_rand = adjusted_rand_score(y, kmeans.labels_)

    # Obtener la inercia
    inertia = kmeans.inertia_

    # Obtener las asignaciones de clusters
    clusters = kmeans.labels_

    return {
        "inertia": inertia,
        "silhouette_score": silhouette,
        "adjusted_rand_score": adjusted_rand,
        "clusters": clusters
    }



# Cargar el dataset de flores Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas reales (para evaluación)

# Llamar a la función con k=3 clusters
resultados = entrenar_y_evaluar_kmeans(X, y, k=3)

# Mostrar los resultados
print("Inercia del modelo:", resultados["inertia"])
print("Silhouette Score:", resultados["silhouette_score"])
print("Adjusted Rand Score:", resultados["adjusted_rand_score"])
print("Clusters asignados:\n", resultados["clusters"][:10])  # Mostrar primeras 10 asignaciones