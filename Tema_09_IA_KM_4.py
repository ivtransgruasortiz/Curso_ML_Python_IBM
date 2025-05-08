import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


class SleepProfile:
    def __init__(self, duracion, latencia, despertares, variabilidad):
        self.duracion = duracion
        self.latencia = latencia
        self.despertares = despertares
        self.variabilidad = variabilidad

    def to_vector(self):
        return np.array([self.duracion, self.latencia, self.despertares, self.variabilidad])


class SleepDatasetGenerator:
    def __init__(self, n=300):
        self.num_profiles = n

    def generate(self):
            duracion = np.random.normal(7, 1.2, size=self.num_profiles)  # Duración entre 6 y 12 horas
            latencia = np.abs(np.random.normal(20, 10, size=self.num_profiles))  # Latencia entre 1 y 4 horas
            despertares = np.random.poisson(1.5, size=self.num_profiles)  # Despertares entre 1 y 3 veces
            variabilidad = np.abs(np.random.normal(30, 15, size=self.num_profiles))   # Variabilidad entre 1 y 4 horas

            profiles = [SleepProfile(du, l, de, v) for du, l, de, v in zip(duracion, latencia, despertares, variabilidad)]
            return profiles


class SleepClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.scaler = StandardScaler()

    def fit(self, profiles):
        X = np.array([profile.to_vector() for profile in profiles])
        X_sc = self.scaler.fit_transform(X)
        self.kmeans.fit(X_sc)
        self.labels = self.kmeans.labels_
        return X_sc, self.labels

    def predict(self, duracion, latencia, despertares, variabilidad):
        profile = SleepProfile(duracion, latencia, despertares, variabilidad)
        X = profile.to_vector()
        X_sc = self.scaler.transform([X])
        return self.kmeans.predict(X_sc)[0]

    def get_cluster_centers(self):
        return self.scaler.inverse_transform(self.kmeans.cluster_centers_)


class SleepAnalysisExample:
    def run(self):
        generator = SleepDatasetGenerator(100)
        profiles = generator.generate()

        clusterer = SleepClusterer(n_clusters=3)
        X_sc, labels = clusterer.fit(profiles)

        df = pd.DataFrame([p.to_vector() for p in profiles],
                          columns=["Duracion", "Latencia", "Despertares", "Variabilidad"])
        df["Grupo"] = labels

        print("📌 Centroides de los grupos:")
        centers = clusterer.get_cluster_centers()
        for i, c in enumerate(centers):
            print(
                f"Grupo {i}: Duración={c[0]:.2f}h, Latencia={c[1]:.1f}min, Despertares={c[2]:.1f}, Variabilidad={c[3]:.1f}min")

        colores = ['blue', 'green', 'orange']
        plt.figure(figsize=(8, 6))
        for i in range(clusterer.n_clusters):
            subset = df[df["Grupo"] == i]
            plt.scatter(subset["Duracion"], subset["Variabilidad"],
                        c=colores[i], label=f"Grupo {i}", alpha=0.6)
        plt.xlabel("Duración del sueño (horas)")
        plt.ylabel("Variabilidad en horario de dormir (minutos)")
        plt.title("💤 Agrupación de perfiles de sueño (K-Means)")
        plt.grid(True)
        plt.legend()
        plt.show()


example = SleepAnalysisExample()
example.run()