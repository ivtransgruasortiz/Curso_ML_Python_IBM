import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class App:
    def __init__(self, app_name, downloads, rating, size_mb, reviews, revenue=None):
        self.app_name = app_name
        self.downloads = downloads
        self.rating = rating
        self.size_mb = size_mb
        self.reviews = reviews
        self.revenue = revenue

# class PlayerDataset:
#     def __init__(self, Player):
#         self.Player = Player
#     def get_feature_matrix(self):
#         return [player.to_features()[1:4] for player in self.Player]
#     def get_target_vector(self):
#         return [player.to_features()[-1] for player in self.Player]

class RevenuePredictor:
    lrm = LinearRegression()
    def fit(self, list_app):
        X = [[app.rating, app.size_mb, app.reviews] for app in list_app]
        Y = [[app.revenue] for app in list_app]
        X_train, X_tests, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        return self.lrm.fit(X_train, Y_train)
    def predict(self, new_app):
        return self.lrm.predict([[new_app.rating, new_app.size_mb, new_app.reviews]])[0][0]


# Datos simulados de entrenamiento
training_apps = [
    App("TaskPro", 200, 4.2, 45.0, 1800, 120.0),
    App("MindSpark", 150, 4.5, 60.0, 2100, 135.0),
    App("WorkFlow", 300, 4.1, 55.0, 2500, 160.0),
    App("ZenTime", 120, 4.8, 40.0, 1700, 140.0),
    App("FocusApp", 180, 4.3, 52.0, 1900, 130.0),
    App("BoostApp", 220, 4.0, 48.0, 2300, 145.0),
]

# Creamos y entrenamos el predictor
predictor = RevenuePredictor()
predictor.fit(training_apps)

# Nueva app para predecir
new_app = App("FocusMaster", 250, 4.5, 50.0, 3000)
predicted_revenue = predictor.predict(new_app)

print(f"Ingresos estimados para {new_app.app_name}: ${predicted_revenue:.2f}K")
