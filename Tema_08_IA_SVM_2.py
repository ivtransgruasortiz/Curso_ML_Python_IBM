import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class ProPlayerClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)  # Configuración del modelo SVM, random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, player_stats):
        return self.model.predict([player_stats])[0]

    def evaluate(self, X_test, y_test):
        """
        Evalúa la precisión del modelo sobre un conjunto de prueba.
        """
        predictions = self.model.predict(X_test)
        return accuracy_score(y_test, predictions)


class GameSimulator:
    def __init__(self):
        # Instanciamos el modelo
        self.classifier = ProPlayerClassifier()

    def run(self):
        # Datos de entrenamiento: cada fila representa un jugador
        X_train = [
            [0.85, 0.9, 0.88, 0.2, 0.8],
            [0.60, 0.4, 0.55, 0.6, 0.5],
            [0.90, 0.95, 0.92, 0.15, 0.9],
            [0.45, 0.3, 0.5, 0.7, 0.3],
            [0.88, 0.85, 0.9, 0.25, 0.85],
            [0.40, 0.35, 0.45, 0.8, 0.4],
        ]

        # Etiquetas: 1 = profesional, 0 = casual
        y_train = [1, 0, 1, 0, 1, 0]

        # Entrenamos el modelo con estos datos
        self.classifier.train(X_train, y_train)

        # Jugador nuevo cuyas estadísticas queremos clasificar
        test_player = [0.83, 0.92, 0.86, 0.18, 0.75]
        prediction = self.classifier.predict(test_player)

        print("Jugador profesional:" if prediction == 1 else "Jugador casual.")

        # Evaluación adicional del modelo con datos nuevos
        X_test = [
            [0.86, 0.9, 0.89, 0.19, 0.82],
            [0.50, 0.45, 0.48, 0.65, 0.4],
        ]
        y_test = [1, 0]
        accuracy = self.classifier.evaluate(X_test, y_test)
        print(f"Precisión del modelo: {accuracy:.2f}")


simulator = GameSimulator()
simulator.run()