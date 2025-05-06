import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class PlayerMatchData:
    def __init__(self, kills, deaths, assists, damage_dealt, damage_received, healing_done, objective_time, won=None):
        self.kills = kills
        self.deaths = deaths
        self.assists = assists
        self.damage_dealt = damage_dealt
        self.damage_received = damage_received
        self.healing_done = healing_done
        self.objective_time = objective_time
        self.won = won
    def to_dict(self):
        return {
            'kills': self.kills,
            'deaths': self.deaths,
            'assists': self.assists,
            'damage_dealt': self.damage_dealt,
            'damage_received': self.damage_received,
            'healing_done': self.healing_done,
            'objective_time': self.objective_time
        }

def generate_synthetic_data(n=100):
    data = []
    for _ in range(n):
    # Genera cada variable siguiendo las instrucciones dadas
    # Crea un objeto PlayerMatchData con estos valores
    # Añádelo a la lista de datos
        kills = np.random.poisson(5)
        deaths = np.random.poisson(3)
        assists = np.random.poisson(2)
        damage_dealt = kills * 300 + np.random.normal(0, 100)
        damage_received = deaths * 400 + np.random.normal(0, 100)
        healing_done = np.random.randint(0, 301)
        objective_time = np.random.randint(0, 121)
        won = 1 if (damage_dealt > damage_received) and (kills > deaths) else 0
        data.append(PlayerMatchData(kills, deaths, assists, damage_dealt, damage_received, healing_done, objective_time, won))
    return data

class VictoryPredictor:
    lrm = LogisticRegression()
    def train(self, data):
        # Entrena el modelo con los datos de entrenamiento
        X = [[x.kills, x.deaths, x.assists, x.damage_dealt, x.damage_received, x.healing_done, x.objective_time] for x in data]
        Y = [x.won for x in data]
        X_train, X_tests, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        return self.lrm.fit(X_train, Y_train)

    def predict(self, player):
        # Realiza la prediccion con el jugador de prueba
        X = [player.kills, player.deaths, player.assists, player.damage_dealt, player.damage_received, player.healing_done, player.objective_time]
        prediction = self.lrm.predict([X])[0]
        return prediction


# Crear datos de entrenamiento
training_data = generate_synthetic_data(150)

# Entrenar modelo
predictor = VictoryPredictor()
predictor.train(training_data)

# Crear jugador de prueba
test_player = PlayerMatchData(8, 2, 3, 2400, 800, 120, 90, None)

# Predecir si ganará
prediction = predictor.predict(test_player)
print(f"¿El jugador ganará? {'Sí' if prediction == 1 else 'No'}")