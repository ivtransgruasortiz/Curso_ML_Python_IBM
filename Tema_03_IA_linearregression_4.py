import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



class VehicleRecord:
    def __init__(self, horas_uso, nivel_desgaste=None):
        self.horas_uso = horas_uso
        self.nivel_desgaste = nivel_desgaste

    def to_vector(self):
        return [self.horas_uso]


class VehicleDataGenerator:
    def __init__(self, num_samples=1000):
        self.records = []
        self.num_samples = num_samples

    def generate(self):
        hours = np.random.uniform(50, 500, self.num_samples)
        wear = 10 + 0.18 * hours + np.random.normal(0, 5, self.num_samples)
        wear = np.clip(wear, 0, 100)
        data = [VehicleRecord(h, w) for h, w in zip(hours, wear)]
        return data


class VehicleWearRegressor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, records):
        X = np.array([record.to_vector() for record in records])
        y = np.array([record.nivel_desgaste for record in records])
        self.model.fit(X, y)

    def predict(self, horas):
        return self.model.predict([[horas]])[0]

    def get_model(self):
        return self.model

class VehicleWearPredictionExample:
    def __init__(self):
        self.generator = VehicleDataGenerator(100)
        self.regressor = VehicleWearRegressor()

    def run(self):
        records = self.generator.generate()
        self.regressor.fit(records)
        test_hours = 250
        prediction = self.regressor.predict(test_hours)
        print(f"⏱ Horas de uso estimadas: {test_hours}")
        print(f"⚙️ Nivel de desgaste estimado: {prediction:.2f}%")

        # Visualización
        X = [record.horas_uso for record in records]
        y = [record.nivel_desgaste for record in records]
        X_test = np.linspace(0, 1000, 100).reshape(-1, 1)
        y_test = self.regressor.get_model().predict(X_test)
        plt.scatter(X, y, color='blue')
        plt.plot(X_test, y_test, color='red')
        plt.axvline(test_hours, color='green', linestyle='--')
        plt.xlabel('Horas de uso')
        plt.ylabel('Nivel de desgaste')
        plt.show()


example = VehicleWearPredictionExample()
example.run()