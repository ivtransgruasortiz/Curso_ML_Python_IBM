import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def generate_dataset(n_samples=100, seed=42):
    np.random.seed(seed)
    X = []
    y = []

    for _ in range(n_samples):
        velocidad = np.random.rand()  # ¿Importa que el lenguaje sea rápido?
        mantenimiento = np.random.rand()  # ¿Fácil de mantener?
        libs = np.random.rand()  # ¿Cuántas librerías se necesitan?
        tipo_app = np.random.randint(0, 3)  # 0 = científica, 1 = móvil/web, 2 = embebida
        rendimiento = np.random.rand()  # ¿Nivel de exigencia de rendimiento?

        # Lógica para asignar lenguaje
        if rendimiento > 0.8 and tipo_app == 2:
            lenguaje = 3  # C++
        elif mantenimiento > 0.7 and tipo_app == 1:
            lenguaje = 2  # Java
        elif libs > 0.6 and tipo_app == 0:
            lenguaje = 0  # Python
        else:
            lenguaje = 1  # JavaScript

        X.append([velocidad, mantenimiento, libs, tipo_app, rendimiento])
        y.append(lenguaje)

    return np.array(X), np.array(y)


class LanguagePredictor:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.label_map = {
            0: "Python",
            1: "JavaScript",
            2: "Java",
            3: "C++"
        }

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, features):
        prediction = self.model.predict([features])[0]
        return self.label_map[prediction]


# Generar datos y entrenar
X, y = generate_dataset()
predictor = LanguagePredictor()
predictor.train(X, y)

# Crear un proyecto nuevo
new_project = np.array([0.7, 0.9, 0.5, 1, 0.6])  # Características del proyecto

# Predecir lenguaje ideal
pred = predictor.predict(new_project)
print(f"Lenguaje recomendado para el nuevo proyecto: {pred}")