import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Player:
    def __init__(self, name, avg_session_time, avg_actions_per_min, avg_kills_per_session, victories=None):
        self.name = name
        self.avg_session_time = avg_session_time
        self.avg_actions_per_min = avg_actions_per_min
        self.avg_kills_per_session = avg_kills_per_session
        self.victories = victories
    def to_features(self):
        return [self.name, self.avg_session_time, self.avg_actions_per_min, self.avg_kills_per_session, self.victories]

class PlayerDataset:
    def __init__(self, Player):
        self.Player = Player
    def get_feature_matrix(self):
        return [player.to_features()[1:4] for player in self.Player]
    def get_target_vector(self):
        return [player.to_features()[-1] for player in self.Player]

class VictoryPredictor:
    lrm = LinearRegression()
    def train(self, dataset):
        X = dataset.get_feature_matrix()
        Y = dataset.get_target_vector()
        X_train, X_tests, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        return self.lrm.fit(X_train, Y_train)
    def predict(self, player):
        return self.lrm.predict([player.to_features()[1:4]])[0]


players = [
    Player("Alice", 40, 50, 6, 20),
    Player("Bob", 30, 35, 4, 10),
    Player("Charlie", 50, 60, 7, 25),
    Player("Diana", 20, 25, 2, 5),
    Player("Eve", 60, 70, 8, 30)
]

dataset = PlayerDataset(players)
predictor = VictoryPredictor()
predictor.train(dataset)

predictor.train(dataset)

test_player = Player("TestPlayer", 45, 55, 5)
predicted = predictor.predict(test_player)
print(f"Victorias predichas para {test_player.name}: {predicted:.2f}")
