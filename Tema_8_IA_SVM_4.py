import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class Piece:
    def __init__(self, textura, simetria, bordes, offset, label=None):
        self.textura = textura
        self.simetria = simetria
        self.bordes = bordes
        self.offset = offset
        self.label = label

    def to_vector(self):
        return [self.textura, self.simetria, self.bordes, self.offset]

class PieceDatasetGenerator:
    def __init__(self, n=400):
        self.num_samples = n  # Número de ejemplos de entrenamiento

    def generate(self):
        pieces = []
        for _ in range(self.num_samples):
            textura = np.random.normal(0.5, 0.15)
            simetria = np.random.normal(0.6, 0.2)
            bordes = np.random.normal(50, 15)
            offset = np.abs(np.random.normal(0, 0.2))
            if (((simetria < 0.4) and
                 (offset > 0.25)) or
                    ((textura < 0.35) or
                     (bordes < 30) or
                     (offset > 0.35))):
                label = "Defectuosa"
            else:
                label = "Correcta"
            piece = Piece(textura, simetria, bordes, offset, label)
            pieces.append(piece)

        return pieces


class PieceClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)

    def fit(self, pieces):
        X = np.array([piece.to_vector() for piece in pieces])
        y = np.array([piece.label for piece in pieces])
        self.model.fit(X, y)

    def predict(self, textura, simetria, bordes, offset):
        return self.model.predict([[textura, simetria, bordes, offset]])[0]

    def evaluate(self, pieces):
        X = np.array([piece.to_vector() for piece in pieces])
        y = np.array([piece.label for piece in pieces])
        y_pred = self.model.predict(X)
        print(confusion_matrix(y, y_pred))
        print(classification_report(y, y_pred))

class PieceAnalysisExample:
    def run(self):
        dataset_generator = PieceDatasetGenerator(400)
        pieces = dataset_generator.generate()
        train_pieces, test_pieces = train_test_split(pieces, test_size=0.3, random_state=42)
        classifier = PieceClassifier()
        classifier.fit(train_pieces)
        classifier.evaluate(test_pieces)

        # PREDICCION NUEVA
        textura = 0.45
        simetria = 0.50
        bordes = 45
        offset = 0.15
        prediction = classifier.predict(textura, simetria, bordes, offset)

        print("🔍 Predicción para condiciones nuevas:")
        print(f"   Textura: {textura}")
        print(f"   Simetria: {simetria}")
        print(f"   Bordes:: {bordes}")
        print(f"   Offset: {offset}")
        print(f"   Etiqueta: {prediction}")

        df = pd.DataFrame({
            "Textura": [r.textura for r in pieces],
            "Simetria": [r.simetria for r in pieces],
            "Bordes": [r.bordes for r in pieces],
            "Offset": [r.offset for r in pieces],
            "Etiqueta": [r.label for r in pieces]
        })

        colores = {
            "Correcta": "green",
            "Defectuosa": "red",
        }

        plt.figure(figsize=(10, 6))
        for etiqueta, color in colores.items():
            plt.scatter(df[df["Etiqueta"] == etiqueta]["Textura"], df[df["Etiqueta"] == etiqueta]["Simetria"], c=color, label=etiqueta)
        plt.xlabel("Textura")
        plt.ylabel("Simetria")
        plt.title("Simetria vs Textura")


example = PieceAnalysisExample()
example.run()