import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Simulación de datos
def generar_datos_compras(num_muestras):
    caracteristicas = []
    etiquetas = []
    for _ in range(num_muestras):
        num_paginas_vistas = np.random.randint(1, 21)
        tiempo_en_sitio = round(np.random.uniform(1, 30), 2)
        if (num_paginas_vistas > 5) and (tiempo_en_sitio > 10):
            etiqueta = 1
        elif (num_paginas_vistas <= 5) and (tiempo_en_sitio <= 10):
            etiqueta = 0
        else:
            etiqueta = 0
        caracteristicas.append([num_paginas_vistas, tiempo_en_sitio])
        etiquetas.append(etiqueta)
    return np.array(caracteristicas), np.array(etiquetas)

def entrenar_modelo(caracteristicas, etiquetas):
    X = caracteristicas
    y = etiquetas
    model = LogisticRegression()
    model.fit(X, y)
    return model

def predecir_compra(modelo, num_paginas_vistas, tiempo_en_sitio):
    caracteristicas = [[num_paginas_vistas, tiempo_en_sitio]]
    prediccion = modelo.predict(caracteristicas)[0]
    return 'El usuario no comprará el producto.' if prediccion == 0 else 'El usuario comprara el producto.'

def evaluar_modelo(modelo, caracteristicas, etiquetas):
    X = caracteristicas
    y = etiquetas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

datos = generar_datos_compras(100)
modelo = entrenar_modelo(datos[0], datos[1])
precision = evaluar_modelo(modelo, datos[0], datos[1])
predecir_compra(modelo, 8, 12)
