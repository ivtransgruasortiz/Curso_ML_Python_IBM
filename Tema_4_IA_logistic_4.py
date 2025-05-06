import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


class WeatherRecord:
    def __init__(self, humedad, presion, lluvia=None):
        self.humedad = humedad
        self.presion = presion
        self.lluvia = lluvia

    def to_vector(self):
        return [self.humedad, self.presion]

class WeatherDataGenerator:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def generate(self):
        humedad = np.round(np.random.uniform(0, 100, self.num_samples), 2)
        presion = np.round(np.random.normal(1000, 50, self.num_samples), 2)

        rain_prob = (humedad - 50) * 0.03 - (presion - 1010) * 0.02
        rain_prob = 1 / (1 + np.exp(-rain_prob))
        rain = (rain_prob > 0.5).astype(int)

        data = [WeatherRecord(h, p, r) for h, p, r in zip(humedad, presion, rain)]
        return data


class WeatherRainClassifier:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, records):
        X = np.array([record.to_vector() for record in records])
        y = np.array([record.lluvia for record in records])
        self.model.fit(X, y)

    def predict(self, humedad, presion):
        return self.model.predict([[humedad, presion]])[0]

    def get_model(self):
        return self.model

    def evaluate(self, records):
        X = np.array([record.to_vector() for record in records])
        y = np.array([record.lluvia for record in records])
        y_pred = self.model.predict(X)
        print(confusion_matrix(y, y_pred))
        print(classification_report(y, y_pred))


class WeatherRainPredictionExample:
    def run(self):
        generator = WeatherDataGenerator(num_samples=1000)
        records = generator.generate()
        train_records, test_records = train_test_split(records, test_size=0.3, random_state=42)

        classifier = WeatherRainClassifier()
        classifier.fit(train_records)
        classifier.evaluate(test_records)

        # PREDICCIÓN NUEVA
        humidity_test = 80
        pressure_test = 995
        prediction = classifier.predict(humidity_test, pressure_test)

        print("🔍 Predicción para condiciones nuevas:")
        print(f"   Humedad: {humidity_test}%")
        print(f"   Presión: {pressure_test} hPa")
        print(f"   ¿Lloverá?: {'Sí ☔' if prediction == 1 else 'No ☀️'}")

        df = pd.DataFrame({
            "Humedad": [r.humedad for r in test_records],
            "Presión": [r.presion for r in test_records],
            "Lluvia": [r.lluvia for r in test_records]
        })

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(df["Humedad"], df["Presión"], c=df["Lluvia"], cmap="bwr", alpha=0.6)
        plt.xlabel("Humedad relativa (%)")
        plt.ylabel("Presión atmosférica (hPa)")
        plt.title("🌧️ Predicción de lluvia según condiciones meteorológicas")
        plt.grid(True)
        plt.colorbar(scatter, label="0 = No llueve, 1 = Llueve")
        plt.show()

WeatherRainPredictionExample().run()