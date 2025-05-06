import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def entrenar_y_evaluar_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matriz_confusion = confusion_matrix(y_test, y_pred)
    reporte = classification_report(y_test, y_pred)
    return {"predicciones": y_pred, "accuracy": accuracy, "matriz_confusion": matriz_confusion, "reporte": reporte}

# Cargar el dataset de vinos
wine = load_wine()
X = wine.data  # Características
y = wine.target  # Clases de vinos

# Dividir en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Llamar a la función y obtener las métricas
resultados = entrenar_y_evaluar_random_forest(X_train, y_train, X_test, y_test)

# Mostrar los resultados
print("Precisión del modelo:", resultados["accuracy"])
print("Matriz de Confusión:\n", resultados["matriz_confusion"])
print("Reporte de Clasificación:\n", resultados["reporte"])