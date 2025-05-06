import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Función de regresión logística
def regresion_logistica(datos):
    X = datos[["Edad", "Colesterol"]]
    Y = datos[["Enfermedad"]]
    X_train, X_tests, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    lrm = LogisticRegression()
    return lrm.fit(X_train, Y_train)



# Ejemplo de uso con datos de pacientes
data = {
    'Edad': [50, 35, 65, 28, 60],
    'Colesterol': [180, 150, 210, 130, 190],
    'Enfermedad': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
modelo_regresion_logistica = regresion_logistica(df)

# Estimaciones de clasificación binaria para nuevos datos
nuevos_datos = pd.DataFrame({'Edad': [45, 55], 'Colesterol': [170, 200]})
estimaciones_clasificacion = modelo_regresion_logistica.predict(nuevos_datos)
print("Estimaciones de Clasificación:")
print(estimaciones_clasificacion)
