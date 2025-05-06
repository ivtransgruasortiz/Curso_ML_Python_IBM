import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.constants import calorie

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class BasketballPlayer:
    def __init__(self, height, weight, avg_points, performance=None):
        self.height = height
        self.weight = weight
        self.avg_points = avg_points
        self.performance = performance  # 0: Low, 1: Medium, 2: High

    def to_vector(self):
        return [self.height, self.weight, self.avg_points]

class BasketballDataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        records = []
        for _ in range(self.num_samples):
            height = np.random.randint(180, 225)
            weight = np.random.randint(90, 130)
            avg_points = np.round(np.random.normal(10, 5))
            if avg_points < 8:
                performance = "Bajo"
            elif (avg_points >= 8) & (avg_points <= 15):
                performance = "Medio"
            else:
                performance = "Alto"
            records.append(BasketballPlayer(height, weight, max(0, avg_points), performance))
        return records

class BasketballPerformanceClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, players):
        X = np.array([player.to_vector() for player in players])
        y = np.array([player.performance for player in players])
        self.model.fit(X, y)

    def predict(self, altura, peso, promedio_puntos):
        return self.model.predict([[altura, peso, promedio_puntos]])[0]

    def evaluate(self, records):
        X = np.array([record.to_vector() for record in records])
        y = np.array([record.performance for record in records])
        y_pred = self.model.predict(X)
        print(confusion_matrix(y, y_pred))
        print(classification_report(y, y_pred))


class BasketballPredictionExample:
    def run(self):
        generator = BasketballDataGenerator()
        records = generator.generate()
        train_records, test_records = train_test_split(records, test_size=0.2, random_state=42)

        classifier = BasketballPerformanceClassifier()
        classifier.fit(train_records)
        classifier.evaluate(test_records)

        # PREDICCION NUEVA
        height_test = 198
        weight_test = 92
        avg_points_test = 17
        prediction = classifier.model.predict([[height_test, weight_test, avg_points_test]])[0]

        print("🔍 Predicción para condiciones nuevas:")
        print(f"   Altura: {height_test} cm")
        print(f"   Peso: {weight_test} kg")
        print(f"   Promedio de puntos por partido: {avg_points_test}")
        print(f"   Rendimiento: {prediction}")

        colores = {
            "Bajo": "red",
            "Medio": "orange",
            "Alto": "green"
        }

        df = pd.DataFrame({
            "Altura": [r.height for r in records],
            "Prom. Puntos": [r.avg_points for r in records],
            "Rendimiento": [r.performance for r in records]
        })

        colores = {
            "Bajo": "red",
            "Medio": "orange",
            "Alto": "green"
        }

        plt.figure(figsize=(8, 6))
        for nivel, color in colores.items():
            subset = df[df["Rendimiento"] == nivel]
            plt.scatter(subset["Altura"], subset["Prom. Puntos"], label=nivel, c=color, alpha=0.6)

        plt.xlabel("Altura (cm)")
        plt.ylabel("Promedio de puntos por partido")
        plt.title("🏀 Clasificación de jugadores de baloncesto por rendimiento")
        plt.grid(True)
        plt.legend(title="Rendimiento")
        plt.show()


BasketballPredictionExample().run()
