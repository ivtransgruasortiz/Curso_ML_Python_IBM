import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test):
    model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)  # Configuración del modelo SVM, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matriz_confusion = confusion_matrix(y_test, y_pred)
    reporte = classification_report(y_test, y_pred)
    return {"predicciones": y_pred, "accuracy": accuracy, "matriz_confusion": matriz_confusion, "reporte": reporte}




# Cargar el dataset de dígitos escritos a mano
digits = load_digits()
X = digits.data  # Características (matriz de píxeles)
y = digits.target  # Etiquetas (números del 0 al 9)

# Dividir en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Llamar a la función y obtener las métricas
resultados = entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test)

# Mostrar los resultados
print("Precisión del modelo:", resultados["accuracy"])
print("Matriz de Confusión:\n", resultados["matriz_confusion"])
print("Reporte de Clasificación:\n", resultados["reporte"])