from multiprocessing.synchronize import SEM_VALUE_MAX

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC


class Player:
    def __init__(self, player_name, character_type, avg_session_time, matches_played, aggressive_actions, defensive_actions, items_bought, victories, style=None):
        self.player_name = player_name
        self.character_type = character_type
        self.avg_session_time = avg_session_time
        self.matches_played = matches_played
        self.aggressive_actions = aggressive_actions
        self.defensive_actions = defensive_actions
        self.items_bought = items_bought
        self.victories = victories
        self.style = style

    def to_dict(self):
        return {
            "player_name": self.player_name,
            "character_type": self.character_type,
            "avg_session_time": self.avg_session_time,
            "matches_played": self.matches_played,
            "aggressive_actions": self.aggressive_actions,
            "defensive_actions": self.defensive_actions,
            "items_bought": self.items_bought,
            "victories": self.victories,
            "style": self.style
        }

class GameModel:
    def __init__(self, players_data):
        data = [p.to_dict() for p in players_data]
        self.df = pd.DataFrame(data)

        # Codificación de variables categóricas
        self.label_encoder = LabelEncoder()
        self.df["character_type_enc"] = self.label_encoder.fit_transform(self.df["character_type"])
        self.df["style_enc"] = self.df["style"].map({"aggressive": 0, "strategic": 1})

        # Definir columnas para los modelos
        self.feature_cols = [
            "character_type_enc", "avg_session_time", "matches_played",
            "aggressive_actions", "defensive_actions", "items_bought"
        ]
        self.classifier = None
        self.regressor = None
        self.cluster_model = None

    def train_classification_model(self):
        X = self.df[self.feature_cols]
        y = self.df["style_enc"]
        self.classifier = LogisticRegression()
        self.classifier.fit(X, y)

    def train_regression_model(self):
        X = self.df[self.feature_cols]
        y = self.df["victories"]
        self.regressor = LinearRegression()
        self.regressor.fit(X, y)

    def train_clustering_model(self, n_clusters=3):
        X = self.df[self.feature_cols]
        self.cluster_model = KMeans(n_clusters=n_clusters)
        self.cluster_model.fit(X)

    def predict_style(self, player):
        X = self._prepare_input(player)
        return "aggressive" if self.classifier.predict(X)[0] == 0 else "strategic"

    def predict_victories(self, player):
        X = self._prepare_input(player)
        return self.regressor.predict(X)[0]

    def assign_cluster(self, player):
        X = self._prepare_input(player)
        return self.cluster_model.predict(X)[0]

    def _prepare_input(self, player):
        data = pd.DataFrame([player.to_dict()])
        data["character_type_enc"] = self.label_encoder.fit_transform(data["character_type"])
        X = data[self.feature_cols]
        return X


# Crear datos de prueba para varios jugadores
players_data = [
    Player("P1", "mage", 40, 30, 90, 50, 20, 18, "aggressive"),
    Player("P2", "tank", 60, 45, 50, 120, 25, 24, "strategic"),
    Player("P3", "archer", 50, 35, 95, 60, 22, 20, "aggressive"),
    Player("P4", "tank", 55, 40, 60, 100, 28, 22, "strategic"),
]

# Instanciar el modelo con los datos de los jugadores
model = GameModel(players_data)

# Entrenar los modelos
model.train_classification_model()
model.train_regression_model()
model.train_clustering_model()

# Crear un nuevo jugador para realizar predicciones
new_player = Player("TestPlayer", "mage", 42, 33, 88, 45, 21, 0)

# Realizar predicciones
predicted_style = model.predict_style(new_player)
predicted_victories = model.predict_victories(new_player)
predicted_cluster = model.assign_cluster(new_player)

# Imprimir los resultados de las predicciones
print(f"Estilo de juego predicho para {new_player.player_name}: {predicted_style}")
print(f"Victorias predichas para {new_player.player_name}: {predicted_victories:.2f}")
print(f"Cluster asignado a {new_player.player_name}: {predicted_cluster}")