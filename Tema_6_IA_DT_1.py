import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def entrenar_arbol_decision(X_train, y_train, X_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# Cargar el dataset de flores Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Clases de las flores (Setosa, Versicolor, Virginica)

# Dividir en conjunto de entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Llamar a la función que debes implementar
predicciones = entrenar_arbol_decision(X_train, y_train, X_test)

# Mostrar algunas predicciones
print("Predicciones del Árbol de Decisión:", predicciones[:10])
print("Valores reales:                    ", y_test[:10])