def operaciones_matematicas(a, b):
    suma = a + b
    resta = a - b
    multiplicacion = a * b
    division = a / b if b != 0 else print("División por cero no permitida")
    resto = a % b if b != 0 else print("División por cero no permitida")
    return (suma, resta, multiplicacion, division, resto)


a = 10
b = 3
resultado = operaciones_matematicas(a, b)
print("Suma:", resultado[0])  # 13
print("Resta:", resultado[1])  # 7
print("Multiplicación:", resultado[2])  # 30
print("División:", resultado[3])  # 3.3333...
print("Residuo:", resultado[4])  # 1

#####

def calcular_suma_y_promedio(lista_numeros):
    if (lista_numeros != None) and (lista_numeros != []) and (lista_numeros != 0):
        suma = sum(lista_numeros)
        promedio = suma / len(lista_numeros)
    else:
        suma = 0
        promedio = 0
    resultado ={"suma": suma, "promedio": promedio}
    # promedio = suma / len(lista_numeros)
    return resultado

numeros = [1, 2, 3, 4, 5]
resultado = calcular_suma_y_promedio(numeros)
print("Suma:", resultado["suma"])
print("Promedio:", resultado["promedio"])

####

def contar_frecuencia(lista):
    dist = list(set(lista))
    dicc = {}
    for i in dist:
        freq = len([x for x in lista if x == i])
        dicc.update({i: freq})
    return dicc


elementos = [1, 2, 2, 3, 1, 2, 4, 5, 4]
resultado = contar_frecuencia(elementos)
print(resultado)

####

def aplicar_funcion_y_filtrar(lista, valor_umbral):
    # lista_cuad = [x**2 for x in lista]
    # lista_final = list(filter(lambda x: x if x > valor_umbral else None, lista_cuad))
    lista_final = [x ** 2 for x in lista if x ** 2 > valor_umbral]

    return lista_final


numeros = [1, 2, 3, 4, 5]
valor_umbral = 3
resultado = aplicar_funcion_y_filtrar(numeros, valor_umbral)
print(resultado)

