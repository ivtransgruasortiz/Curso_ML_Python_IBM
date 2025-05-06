import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# Función de clasificación KNN
def knn_clasificacion(datos, k=3):
    X = datos[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    Y = datos["species"]
    knn = KNeighborsClassifier(n_neighbors=k)
    return knn.fit(X, Y)


# Ejemplo de uso con el conjunto de datos Iris
# data = pd.read_csv('Iris.csv')

# path = "file://localhost/Users/iortiz/Desktop/CURSOS_KNOWMADMOOD/Curso_IA_IBM/pycharm_projects_IA_ML/Iris.csv"
# url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# url = "https://github.com/ivtransgruasortiz/datasets/blob/master/Iris.csv"
url = "https://raw.githubusercontent.com/ivtransgruasortiz/datasets/refs/heads/master/Iris2.csv"

data = (pd.read_csv(url)
        .rename({"sepal.length": "sepal_length",
                 "sepal.width": "sepal_width",
                 "petal.length": "petal_length",
                 "petal.width": "petal_width",
                 "variety": "species"},
                axis=1))
# Reemplaza 'iris.csv' con tu archivo de datos
modelo_knn = knn_clasificacion(data, k=3)

# Estimaciones de clasificación para nuevas muestras
nuevas_muestras = pd.DataFrame({
    'sepal_length': [5.1, 6.0, 4.4],
    'sepal_width': [3.5, 2.9, 3.2],
    'petal_length': [1.4, 4.5, 1.3],
    'petal_width': [0.2, 1.5, 0.2]
})

estimaciones_clasificacion = modelo_knn.predict(nuevas_muestras)
print("Estimaciones de Clasificación:")
print(estimaciones_clasificacion)