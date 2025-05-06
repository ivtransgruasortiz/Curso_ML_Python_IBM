import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class App:
    def __init__(self, app_name, monthly_users, avg_session_length, retention_rate, social_shares, success=None):
        self.app_name = app_name
        self.monthly_users = monthly_users
        self.avg_session_length = avg_session_length
        self.retention_rate = retention_rate
        self.social_shares = social_shares
        self.success = success
    def to_features(self):
        return [self.monthly_users, self.avg_session_length, self.retention_rate, self.social_shares, self.success]

class AppDataset:
    def __init__(self, app):
        self.App = app
    def get_feature_matrix(self):
        return [app.to_features()[:4] for app in self.App]
    def get_target_vector(self):
        return [app.to_features()[-1] for app in self.App]

class SuccessPredictor:
    lrm = LogisticRegression()
    def train(self, dataset):
        X = dataset.get_feature_matrix()
        Y = dataset.get_target_vector()
        X_train, X_tests, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        return self.lrm.fit(X_train, Y_train)
    def predict(self, new_app):
        return self.lrm.predict([new_app.to_features()[:4]])[0]
    def predict_proba(self, new_app):
        return self.lrm.predict_proba([new_app.to_features()[:4]])[0][1]


# Datos de entrenamiento
apps = [
    App("FastChat", 10000, 12.5, 0.65, 1500, 1),
    App("FitTrack", 500, 5.0, 0.2, 50, 0),
    App("GameHub", 15000, 25.0, 0.75, 3000, 1),
    App("BudgetBuddy", 800, 6.5, 0.3, 80, 0),
    App("EduFlash", 12000, 18.0, 0.7, 2200, 1),
    App("NoteKeeper", 600, 4.0, 0.15, 30, 0)
]


dataset = AppDataset(apps)
predictor = SuccessPredictor()
predictor.train(dataset)

# Nueva app a evaluar
new_app = App("StudyBoost", 20000, 15.0, 0.5, 700)
predicted_success = predictor.predict(new_app)
prob = predictor.predict_proba(new_app)

print(f"¿Será exitosa la app {new_app.app_name}? {'Sí' if predicted_success else 'No'}")
print(f"Probabilidad estimada de éxito: {prob:.2f}")
predicted_revenue = predictor.predict(new_app)
