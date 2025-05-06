import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import weibull
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class RecyclableItem:
    def __init__(self, weight, volume, material_type=None):
        self.weight = weight
        self.volume = volume
        self.material_type = material_type

    def to_vector(self):
        return [self.weight, self.volume]


class RecyclingDataGenerator:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def generate(self):
        items  = []
        for _ in range(self.num_samples):
            weight = np.round(np.random.uniform(40, 220 ), 2)
            volume = np.round(np.random.uniform(70, 160), 2)
            if (weight >= 80) and (weight <= 120) and (volume >= 90) and (volume <= 130):
                material_type = 0
            elif (weight >= 180) and (weight <= 220) and (volume >= 70) and (volume <= 110):
                material_type = 1
            elif (weight >= 40) and (weight <= 70) and (volume >= 120) and (volume <= 160):
                material_type = 2
            else:
                material_type = 0
            items.append(RecyclableItem(weight, volume, material_type))
        return items

class RecyclableMaterialClassifier:
    def __init__(self, k):
        self.model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, records):
        X = np.array([record.to_vector() for record in records])
        y = np.array([record.material_type for record in records])
        self.model.fit(X, y)

    def predict(self, peso, volumen):
        return self.model.predict([[peso, volumen]])[0]

    def evaluate(self, records):
        X = np.array([record.to_vector() for record in records])
        y = np.array([record.material_type for record in records])
        y_pred = self.model.predict(X)
        print(confusion_matrix(y, y_pred))
        print(classification_report(y, y_pred))

class RecyclablePredictionExample:
    def run(self):
        generator = RecyclingDataGenerator(num_samples=1000)
        records = generator.generate()
        train_records, test_records = train_test_split(records, test_size=0.2, random_state=42)

        classifier = RecyclableMaterialClassifier(k=5)
        classifier.fit(train_records)
        classifier.evaluate(test_records)

        # PREDICCIÓN NUEVA
        weight = 110
        volume = 105
        prediction = classifier.predict(weight, volume)

        print("🔍 Predicción para condiciones nuevas:")
        print(f"   Peso: {weight} kg")
        print(f"   Volumen: {volume} cm3")
        print(f"   Tipo de material: {prediction}")

        df = pd.DataFrame({
            "Peso": [r.weight for r in records],
            "Volumen": [r.volume for r in records],
            "Material": [r.material_type for r in records]
        })

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(df["Peso"], df["Volumen"], c=df["Material"], cmap="bwr", alpha=0.6)
        plt.xlabel("Peso (kg)")
        plt.ylabel("Volumen (cm3)")
        plt.title("🌧️ Predicción de material según peso y volumen")
        plt.colorbar(scatter, label="Tipo de material")
        plt.show()

RecyclablePredictionExample().run()
