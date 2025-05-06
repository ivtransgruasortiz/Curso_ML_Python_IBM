####
#### matplotlib ####
####

import matplotlib.pyplot as plt

def graficar_linea(x, y):
    plt.scatter(x, y)
    plt.title("Relación entre horas de estudio y calificaciones")
    plt.xlabel("Horas de Estudio")
    plt.ylabel("Calificación")
    plt.show()

x = [1, 2, 3, 4, 5, 6, 7, 8]

y = [55, 60, 65, 70, 75, 80, 85, 90]

graficar_linea(x, y)

####

import matplotlib.pyplot as plt


def graficar_temperaturas(dias, temperaturas):
    plt.plot(dias, temperaturas, "b--")
    plt.title("Temperaturas Semanales")
    plt.xlabel("Días")
    plt.ylabel("Temperatura (°C)")
    plt.show()


dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
temperaturas = [22, 24, 23, 25, 26, 28, 27]

graficar_temperaturas(dias, temperaturas)

####

