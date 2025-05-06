import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class AirSample:
    def __init__(self, pm25, pm10, o3, no2, quality=None):
        self.pm25 = pm25
        self.pm10 = pm10
        self.o3 = o3
        self.no2 = no2
        self.quality = quality  # 0: Saludable, 1: Contaminado

    def to_vector(self):
        return [self.pm25, self.pm10, self.o3, self.no2]

class AirDataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        samples = []
        for _ in range(self.num_samples):
            pm25 = np.random.uniform(0, 150)
            pm10 = np.random.uniform(0, 200)
            o3 = np.random.uniform(0, 120)
            no2 = np.random.uniform(0, 60)
            quality = int((pm25 > 50) or (pm10 > 80) or (o3 > 100) or (no2 > 40)) # 0: Saludable, 1: Contaminado
            sample = AirSample(pm25, pm10, o3, no2, quality)
            samples.append(sample)
        return samples

class AirQualityClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', C=10.0, random_state=42)

    def fit(self, samples):
        X = [sample.to_vector() for sample in samples]
        y = [sample.quality for sample in samples]
        self.model.fit(X, y)

    def predict(self, sample):
        return self.model.predict([sample.to_vector()])[0]

class AirQualityExample:
    def run(self):
        generador = AirDataGenerator(200)
        training_data = generador.generate()
        clasifier = AirQualityClassifier()
        clasifier.fit(training_data)
        new_sample = AirSample(pm25=22, pm10=30, o3=50, no2=35)
        prediction = clasifier.predict(new_sample)
        print(f"🌍 Muestra de aire: PM2.5: {new_sample.pm25}, "
              f"PM10: {new_sample.pm10}, "
              f"O3: {new_sample.o3}, "
              f"NO2: {new_sample.no2}\n"
              f"✅ Predicción de calidad: {'Saludable' if prediction == 0 else 'Contaminado'}")


example = AirQualityExample()
example.run()