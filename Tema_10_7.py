import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def generar_datos_emails(num_muestras):
    caracteristicas = []
    etiquetas = []
    for _ in range(num_muestras):
        longitud_mensaje = np.random.randint(50, 501)
        frecuencia_palabra_clave = round(np.random.rand(), 2)
        cantidad_enlaces = np.random.randint(1, 11)
        if (frecuencia_palabra_clave < 0.5) and ((cantidad_enlaces < 3) or (longitud_mensaje < 300)):
            spam = 0
        elif (frecuencia_palabra_clave >= 0.5) and ((cantidad_enlaces >= 3) or (longitud_mensaje >= 300)):
            spam = 1
        else:
            spam = 1
        caracteristicas.append([longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces])
        etiquetas.append(spam)
    return np.array(caracteristicas), np.array(etiquetas)

def entrenar_modelo_svm(datos, etiquetas):
    X = datos
    y = etiquetas
    model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    model.fit(X, y)
    return model

def predecir_email(modelo, longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces):
    caracteristicas = [[longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces]]
    prediccion = modelo.predict(caracteristicas)[0]
    return 'El email es Spam ' if prediccion == 1 else 'El email no es Spam'

datos = generar_datos_emails(100)
modelo = entrenar_modelo_svm(datos[0], datos[1])
predecir_email(modelo, 8, 12, 4)
