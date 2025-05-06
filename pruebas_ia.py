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
