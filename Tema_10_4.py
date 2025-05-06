import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier


def generar_series(num_series):
    series = []
    for _ in range(num_series):
        serie = sorted(np.random.choice(range(1, 50, 1), 6).tolist())
        series.append(serie)
    return series

def entrenar_modelo():
    X = generar_series(1000)
    y = [np.random.choice([0,1], p=[0.9, 0.1]) for x in range(len(X))]
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X)
    # Entrenar el modelo de clustering
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model.fit(X_transformed, y)

def predecir_mejor_serie(modelo, num_series):
    series_df  = pd.DataFrame(generar_series(num_series))
    scaler = StandardScaler()
    series_transformed = scaler.fit_transform(series_df)
    probabilidad = modelo.predict_proba(series_transformed)[:,1]
    mejor_serie = series_df.iloc[np.argmax(probabilidad)].tolist()
    probabilidad_serie = np.max(probabilidad)
    print("La mejor serie es:", mejor_serie)
    return mejor_serie, probabilidad_serie


modelo = entrenar_modelo()
mejor_serie, probabilidad = predecir_mejor_serie(modelo, 10)
