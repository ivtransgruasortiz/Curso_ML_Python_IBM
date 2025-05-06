import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def generar_datos_frutas(num_muestras):
    # Generar datos de frutas
    características = []
    etiquetas = []
    for _ in range(1, num_muestras+1):
        peso = np.round(np.random.uniform(100, 250), 2)
        tamano = np.round(np.random.uniform(7, 20), 2)
        if (((peso > 120) and (peso < 200)) and
                ((tamano > 7) and (tamano < 9))):
            etiqueta = "Manzana"
        elif (((peso > 100) and (peso < 150)) and
                  ((tamano > 12) and (tamano < 20))):
            etiqueta = "Plátano"
        elif(((peso > 150) and (peso < 250)) and
             ((tamano > 8) and (tamano < 12))):
            etiqueta = "Naranja"
        else:
            etiqueta = "Naranja"
        características.append([peso, tamano])
        etiquetas.append(etiqueta)
    return características, etiquetas

def entrenar_modelo(data):
    # Entrenar el modelo de clustering
    model = KNeighborsClassifier(n_neighbors=3)
    X = [x for x in data[0]]
    y = [x for x in data[1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model

def predecir_fruta(modelo, peso, tamano):
    # Predecir la fruta basado en el modelo
    return modelo.predict([[peso, tamano]])[0]

data = generar_datos_frutas(200)
modelo = entrenar_modelo(data)
predecir_fruta(modelo, 100, 9)