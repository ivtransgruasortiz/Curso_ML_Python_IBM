import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class Individual:
    def __init__(self, heart_rate, cortisol_level, skin_conductance, stress_level=None):
        self.heart_rate = heart_rate
        self.cortisol_level = cortisol_level
        self.skin_conductance = skin_conductance
        self.stress_level = stress_level

    def to_vector(self):
        return [self.heart_rate, self.cortisol_level, self.skin_conductance]


class StressDataGenerator:
    def __init__(self, n=100):
        self.num_samples = n

    def generate(self):
        stress_data = []
        for _ in range(self.num_samples):
            heart_rate = np.round(np.random.normal(75, 15), 2)
            cortisol_level = np.round(np.random.normal(12, 4), 2)
            skin_conductance = np.round(np.random.normal(5, 1.5), 2)
            if (heart_rate > 90) or (cortisol_level > 18) or (skin_conductance > 6.5):
                stress_level = "Alto"
            elif (heart_rate > 70) or (cortisol_level > 10) or (skin_conductance > 4.5):
                stress_level = "Moderado"
            else:
                stress_level = "Bajo"
            stress_data.append(Individual(max(0, heart_rate), max(0, cortisol_level), max(0, skin_conductance), stress_level))
        return stress_data

class StressClassifier:
    def __init__(self, n=100):
        self.model = RandomForestClassifier(n_estimators=n, random_state=42)
        # self.label_map = {
        #     0: "Bajo",
        #     1: "Moderado",
        #     2: "Alto"
        # }

    def fit(self, individuals):
        X = np.array([ind.to_vector() for ind in individuals])
        y = np.array([ind.stress_level for ind in individuals])
        self.model.fit(X, y)

    def predict(self, heart_rate, cortisol, conductance):
        return self.model.predict([[heart_rate, cortisol, conductance]])[0]

    def evaluate(self, test_data):
        X = np.array([ind.to_vector() for ind in test_data])
        y = np.array([ind.stress_level for ind in test_data])
        y_pred = self.model.predict(X)
        print(confusion_matrix(y, y_pred))
        print(classification_report(y, y_pred))


class StressAnalysisExample:
    def run(self):
        stress_generator = StressDataGenerator()
        stress_data = stress_generator.generate()
        train_records, test_records = train_test_split(stress_data, test_size=0.2)

        classifier = StressClassifier(100)
        classifier.fit(train_records)
        classifier.evaluate(test_records)

        # PREDICCION NUEVA
        heart_rate_test = 95
        cortisol_test = 20
        conductance_test = 7
        prediction = classifier.predict(heart_rate_test, cortisol_test, conductance_test)

        print("🔍 Predicción para condiciones nuevas:")
        print(f"   Frecuencia cardiaca: {heart_rate_test} bpm")
        print(f"   Cortisol: {cortisol_test} ng/ml")
        print(f"   Conductancia cutánica: {conductance_test} mS")
        print(f"   Nivel de estres: {prediction}")

        colores = {
            "Bajo": "red",
            "Moderado": "orange",
            "Alto": "green"
        }

        df = pd.DataFrame({
            "Frecuencia cardiaca": [r.heart_rate for r in stress_data],
            "Cortisol": [r.cortisol_level for r in stress_data],
            "Conductancia cutánica": [r.skin_conductance for r in stress_data],
            "Nivel de estres": [r.stress_level for r in stress_data]
        })

        plt.figure(figsize=(8, 6))
        for nivel, color in colores.items():
            subset = df[df["Nivel de estres"] == nivel]
            plt.scatter(subset["Frecuencia cardiaca"], subset["Cortisol"], label=nivel, c=color, alpha=0.6)

        plt.xlabel("Frecuencia cardiaca (bpm)")
        plt.ylabel("Cortisol (ng/ml)")
        plt.title("🧠 Clasificación de individuos por nivel de estres")
        plt.grid(True)
        plt.legend(title="Nivel de estres")
        plt.show()



example = StressAnalysisExample()
example.run()