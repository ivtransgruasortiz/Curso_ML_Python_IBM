import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class Player:
    def __init__(self, name, level, aggressiveness, cooperation, exploration, preferred_class=None):
        self.name = name
        self.level = level
        self.aggressiveness = aggressiveness
        self.cooperation = cooperation
        self.exploration = exploration
        self.preferred_class = preferred_class

    def to_features(self):
        return [self.level, self.aggressiveness, self.cooperation, self.exploration]


class PlayerDataset:
    def __init__(self, player):
        self.player = player
        self.get_x = [player.to_features() for player in self.player]
        self.get_y = [player.preferred_class for player in self.player]


class ClassRecommender:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.trained = False

    def train(self, PlayerDataset):
        self.trained = True
        return self.knn.fit(PlayerDataset.get_x, PlayerDataset.get_y)

    def predict(self, player):
        if not self.trained:
            raise Exception("El modelo no ha sido entrenado.")
        else:
            return self.knn.predict([player.to_features()])[0]

    def get_nearest_neighbors(self, player):
        if not self.trained:
            raise Exception("El modelo no ha sido entrenado.")
        else:
            return self.knn.kneighbors([player.to_features()], return_distance=False)[0]


# Datos de entrenamiento
players = [
    Player("Alice", 20, 0.8, 0.2, 0.1, "Warrior"),
    Player("Bob", 45, 0.4, 0.8, 0.2, "Healer"),
    Player("Cleo", 33, 0.6, 0.4, 0.6, "Archer"),
    Player("Dan", 60, 0.3, 0.9, 0.3, "Healer"),
    Player("Eli", 50, 0.7, 0.2, 0.9, "Mage"),
    Player("Fay", 25, 0.9, 0.1, 0.2, "Warrior"),
]

# Nuevo jugador
new_player = Player("TestPlayer", 40, 0.6, 0.3, 0.8)

# Entrenamiento y predicción
dataset = PlayerDataset(players)
recommender = ClassRecommender(n_neighbors=3)
recommender.train(dataset)

# Resultado
recommended_class = recommender.predict(new_player)
neighbors_indices = recommender.get_nearest_neighbors(new_player)

print(f"Clase recomendada para {new_player.name}: {recommended_class}")
print("Jugadores similares:")
for i in neighbors_indices:
    print(f"- {players[i].name} ({players[i].preferred_class})")

