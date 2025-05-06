import numpy as np

def generar_numeros_enteros_aleatorios(N, minimo, maximo):
    lista = np.random.randint(minimo, maximo, N)
    return lista

N = 5
minimo = 1
maximo = 10
resultado = generar_numeros_enteros_aleatorios(N, minimo, maximo)
print(resultado)

####

import numpy as np

def generar_secuencia_numerica(minimo, maximo, paso):
    lista = np.arange(minimo, maximo, paso)
    return lista


minimo = 0
maximo = 10
paso = 2
resultado = generar_secuencia_numerica(minimo, maximo, paso)
print(resultado)

####

import numpy as np


def analizar_finanzas(ingresos, gastos):
    """
    Analiza las finanzas personales de un estudiante durante un año.
    Calcula el balance mensual, el total de ingresos, el total de gastos
    y el saldo final.

    Args:
    ingresos (np.array): Array de ingresos mensuales (12 elementos).
    gastos (np.array): Array de gastos mensuales (12 elementos).

    Returns:
    np.array: Array con el balance mensual, el total de ingresos,
              el total de gastos y el saldo final.
    """
    ingresos = np.array(ingresos)
    gastos = np.array(gastos)
    balance = ingresos - gastos
    tot_in = ingresos.sum()
    tot_out = gastos.sum()
    saldo = tot_in - tot_out

    return np.array([balance, tot_in, tot_out, saldo])


ingresos = [1500, 1600, 1700, 1650, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
gastos = [1000, 1100, 1200, 1150, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

resultado = analizar_finanzas(ingresos, gastos)
print(resultado)

####
## pandas ##
####

import pandas as pd

def calcular_promedio(dataframe):
    resul = []
    cols = dataframe.columns
    for column in cols:
        print(column)
        resul.append(dataframe[column].mean())
    return pd.Series(resul, cols)

data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}

df = pd.DataFrame(data)
# df = pd.DataFrame([])

resultado = calcular_promedio(df)
print(resultado)

####

import pandas as pd


def seleccionar_datos(dataframe, criterio):
    columna = criterio.split()[0]
    valor = float(criterio.split()[2])
    signo = criterio.split()[1]
    df = eval(f'dataframe[dataframe["{columna}"] {signo} {valor}]')
    return df


data = {
    'nombre': ['Alice', 'Bob', 'Charlie', 'David'],
    'edad': [20, 22, 18, 25],
    'calificaciones': [90, 88, 75, 95]
}

df = pd.DataFrame(data)
dataframe = df
criterio = 'edad > 18'
resultado = seleccionar_datos(df, criterio)
print(resultado)

#####

import pandas as pd


def rellenar_con_media(dataframe, columna):
    media = dataframe[columna].mean()
    dataframe[columna] = dataframe[columna].fillna(media)
    return dataframe


data = {
    'nombre': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'edad': [20, None, 18, 25, None],
    'calificaciones': [90, 88, None, None, 95]
}

df = pd.DataFrame(data)
columna = 'calificaciones'

resultado = rellenar_con_media(df, columna)
print(resultado)

####
