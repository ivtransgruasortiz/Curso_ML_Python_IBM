import numpy as np
from scipy.cluster.vq import kmeans
from sklearn.datasets import load_iris

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


class Traveler:
    def __init__(self, beach, mountain, city, countryside):
        self.beach = beach
        self.mountain = mountain
        self.city = city
        self.countryside = countryside
    def to_features(self):
        return [self.beach, self.mountain, self.city, self.countryside]

class TravelerGenerator :
    def __init__(self, n_samples=200):
        self.n_samples = n_samples
    def generate(self):
        travelers = []
        for _ in range(self.n_samples):
            beach = np.round(np.random.uniform(0, 10), 2)
            mountain = np.round(np.random.uniform(0, 10), 2)
            city = np.round(np.random.uniform(0, 10), 2)
            countryside = np.round(np.random.uniform(0, 10), 2)
            traveler = Traveler(beach, mountain, city, countryside)
            travelers.append(traveler)
        return travelers


class TravelerClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit(self, travelers):
        X = [traveler.to_features() for traveler in travelers]
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.model.fit(X)

    def predict(self, traveler):
        return int(self.model.predict([traveler.to_features()])[0])

    def get_cluster_centers(self):
        return self.model.cluster_centers_

    def print_cluster_summary(self, travelers):
        labels = self.model.predict([traveler.to_features() for traveler in travelers])
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(travelers[i].name)
        for cluster_id, names in clusters.items():
            print(f"🏝️🏔️🏙️🌄 Cluster Centers (Preferencias promedio): {cluster_id}:")
            for name in names:
                print(f"  - {name}")


class TravelerClusteringExample:
    def run(self):
        generador = TravelerGenerator(200)
        training_data = generador.generate()
        clusterer = TravelerClusterer(n_clusters=3)
        clusterer.fit(training_data)
        new_sample = Traveler(8.5, 2.0, 9.0, 1.5)
        prediction = clusterer.predict(new_sample)
        print(f"🔍 Nuevo viajero con preferencias: "
              f"Beach: {new_sample.beach}, "
              f"Mountain: {new_sample.mountain}, "
              f"City: {new_sample.city}, "
              f"Countryside: {new_sample.countryside}\n"
              f"📌 El nuevo viajero pertenece al grupo {prediction}")
        return prediction


# Ejecutar ejemplo
example = TravelerClusteringExample()
example.run()