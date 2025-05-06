import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class IoTKNNClassifier:
    def __init__(self, n_neighbors=3, n_samples=50):
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        paquetes_por_segundo = np.random.randint(10, 1000, self.n_samples)
        bytes_por_paquete = np.random.randint(50, 1500, self.n_samples)
        protocolo = np.random.randint(1, 4, self.n_samples)  # 1 = TCP, 2 = UDP, 3 = HTTP
        seguro = np.random.randint(0, 2, self.n_samples)  # 0 = peligroso, 1 = segur
        self.df = pd.DataFrame({
            "paquetes_por_segundo": paquetes_por_segundo,
            "bytes_por_paquete": bytes_por_paquete,
            "protocolo": protocolo,
            "seguro": seguro
        })
        self.X = self.df.drop(columns=["seguro"])  # Variables independientes
        self.y = self.df["seguro"]  # Variable dependiente (lo que queremos predecir)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.knn = KNeighborsClassifier(self.n_neighbors)

    def train(self):
        """Entrena el modelo k-NN con los datos de entrenamiento."""
        self.knn.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evalúa el modelo con datos de prueba y retorna la precisión."""
        y_pred = self.knn.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def predict(self, nuevo_dispositivo):
        """Predice si un nuevo dispositivo IoT es seguro o peligroso."""
        prediccion = self.knn.predict([nuevo_dispositivo])
        return prediccion[0]