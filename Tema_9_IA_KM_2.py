import numpy as np
from scipy.cluster.vq import kmeans
from sklearn.datasets import load_iris

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


class Player:
    def __init__(self, name, avg_session_time, missions_completed, accuracy, aggressiveness):
        self.name = name
        self.avg_session_time = avg_session_time
        self.missions_completed = missions_completed
        self.accuracy = accuracy
        self.aggressiveness = aggressiveness

    def to_features(self):
        return [self.avg_session_time, self.missions_completed, self.accuracy, self.aggressiveness]


class PlayerClusterer:
    def __init__(self):
        self.model = None
        self.players = []

    def fit(self, players, n_clusters):
        self.players = players
        X = [player.to_features() for player in players]
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.model.fit(X)

    def predict(self, player):
        return int(self.model.predict([player.to_features()])[0])

    def get_cluster_centers(self):
        return self.model.cluster_centers_

    def print_cluster_summary(self, players):
        labels = self.model.predict([player.to_features() for player in players])
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(players[i].name)
        for cluster_id, names in clusters.items():
            print(f"Cluster {cluster_id}:")
            for name in names:
                print(f"  - {name}")


class GameAnalytics:
    def __init__(self):
        self.data = [
            ("Alice", 2.5, 100, 0.85, 0.3),
            ("Bob", 1.0, 20, 0.60, 0.7),
            ("Charlie", 3.0, 150, 0.9, 0.2),
            ("Diana", 0.8, 15, 0.55, 0.9),
            ("Eve", 2.7, 120, 0.88, 0.25),
            ("Frank", 1.1, 30, 0.62, 0.65),
            ("Grace", 0.9, 18, 0.58, 0.85),
            ("Hank", 3.2, 160, 0.91, 0.15)
        ]
        self.players = [Player(*entry) for entry in self.data]
        self.clusterer = PlayerClusterer()

    def run(self):
        self.clusterer.fit(self.players, n_clusters=3)
        self.clusterer.print_cluster_summary(self.players)

        new_player = Player("Zoe", 1.5, 45, 0.65, 0.5)
        cluster = self.clusterer.predict(new_player)
        print(f"\nJugador Zoe pertenece al cluster: {cluster}")

analytics = GameAnalytics()
analytics.run()