import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


class CustomerSegmentationModel:
    """Modelo para segmentar clientes y predecir compras futuras"""

    def __init__(self, df):
        # Ya inicializado para ti
        self.df = df.copy()
        self.data = self.df.copy()
        self.model = None
        self.accuracy = None
        self.conf_matrix = None

    def segment_customers(self, n_clusters=3):
        """
        TODO: Implementa la segmentación de clientes usando K-Means
        Hint: Usa StandardScaler y KMeans
        """

        # TODO: Selecciona las columnas relevantes
        # Hint: ["total_spent", "total_purchases", "purchase_frequency"]
        X = self.data[["total_spent", "total_purchases", "purchase_frequency"]]

        # TODO: Normaliza los datos
        # Hint: Usa StandardScaler()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # TODO: Aplica K-Means
        # Hint: KMeans(n_clusters=n_clusters)
        self.model = KMeans(n_clusters=n_clusters)

        # TODO: Guarda los segmentos en el DataFrame
        self.data["customer_segment"] = self.model.fit_predict(X)
        return self.data

    def train_model(self):
        """
        TODO: Entrena un modelo de regresión logística
        Hint: Usa LogisticRegression y train_test_split
        """
        # TODO: Verifica si existe la segmentación
        if "customer_segment" not in self.data.columns:
            self.segment_customers()

        # TODO: Prepara X (características) e y (objetivo)
        # Hint: X debe incluir total_spent, total_purchases,
        # purchase_frequency y customer_segment
        X = self.data[["total_spent", "total_purchases", "purchase_frequency", "customer_segment"]]
        y = self.data["will_buy_next_month"]

        # TODO: Divide los datos en train y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # TODO: Entrena el modelo y calcula métricas
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.conf_matrix = confusion_matrix(y_test, y_pred)
        pass

    def get_accuracy(self):
        """Ya implementado para ti"""
        return self.accuracy

    def get_confusion_matrix(self):
        """Ya implementado para ti"""
        return self.conf_matrix


### old ####
# import numpy as np
# import pandas as pd
# from pandas.core.common import random_state
#
# from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler
#
# from Tema_3_IA_linearregression_1 import modelo_regresion
# from Tema_6_IA_DT_1 import X_test
#
# """
# class Player:
#     def __init__(self, name, avg_session_time, avg_actions_per_min, avg_kills_per_session, victories=None):
#         self.name = name
#         self.avg_session_time = avg_session_time
#         self.avg_actions_per_min = avg_actions_per_min
#         self.avg_kills_per_session = avg_kills_per_session
#         self.victories = victories
#     def to_features(self):
#         return [self.name, self.avg_session_time, self.avg_actions_per_min, self.avg_kills_per_session, self.victories]
#
# class PlayerDataset:
#     def __init__(self, Player):
#         self.Player = Player
#     def get_feature_matrix(self):
#         return [player.to_features()[1:4] for player in self.Player]
#     def get_target_vector(self):
#         return [player.to_features()[-1] for player in self.Player]
# """
#
#
#
# class SegmentedCustomer:
#     def __init__(self, total_spent, total_purchases, purchase_frequency, will_buy_next_month=None, customer_segment=None):
#         self.total_spent = total_spent
#         self.total_purchases = total_purchases
#         self.purchase_frequency = purchase_frequency
#         self.will_buy_next_month = will_buy_next_month
#         self.customer_segment = customer_segment
#
#     def to_vector(self):
#         return np.array([self.total_spent, self.total_purchases, self.purchase_frequency])
#
#     def to_features(self):
#         return np.array([self.total_spent, self.total_purchases, self.purchase_frequency, self.customer_segment])
#
#
# class CustomerSegmentationGenerator:
#     def __init__(self):
#         self.n_customers = None
#     def generate(self, n_customers=100):
#         self.n_customers = n_customers
#         total_spent = np.random.uniform(100, 5000, self.n_customers)
#         total_purchases = np.random.randint(1, 100, self.n_customers)
#         purchase_frequency = np.random.uniform(1, 30, self.n_customers)
#         will_buy_next_month = np.random.choice([0, 1], self.n_customers, p=[0.7, 0.3])
#         return [SegmentedCustomer(total_spent[i], total_purchases[i], purchase_frequency[i], will_buy_next_month[i]) for i in range(self.n_customers)]
#
# class CustomerSegmentationModel:
#     def __init__(self):
#         self.model = None
#         self.scaler = StandardScaler()
#
#     def fit(self, data):
#         # Aplicar K-Means con 3 clusters
#         X = [x.to_vector() for x in data]
#         X_sc = self.scaler.fit_transform(X)
#         kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
#         self.model = kmeans.fit(X_sc)
#
#     def predict(self, customer):
#         X = [x.to_vector() for x in customer]
#         X_sc = self.scaler.fit_transform(X)
#         return self.model.predict(X_sc)
#
#
# class LogisticRegressionModel:
#     def __init__(self):
#         self.model = None
#
#     def fit(self, data):
#         X = data[['total_spent', 'total_purchases', 'purchase_frequency', 'customer_segment']]
#         y = data['will_buy_next_month']
#         self.X_train, self.X_tests, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#         self.model = LogisticRegression(random_state=42)
#         return self.model.fit(self.X_train, self.y_train)
#
#     def predict(self, customer):
#         X = [x.to_features for x in customer]
#         return self.model.predict(X)[0]
#
#     def evaluate(self):
#         y_pred = self.model.predict(self.X_tests)
#         accuracy = accuracy_score(self.y_test, y_pred)
#         confusion = confusion_matrix(self.y_test, y_pred)
#         print("Accuracy:", accuracy)
#         print("Confusion Matrix:\n", confusion)
#         return accuracy, confusion
#
#
# class CustomerSegmentationExample:
#     def __init__(self):
#         self.generator = CustomerSegmentationGenerator()
#         self.segmentation_model = CustomerSegmentationModel()
#         self.logistic_regression_model = LogisticRegressionModel()  # Agregamos el modelo de regresión logística
#
#     def run(self):  # Agregamos la función run
#         data = self.generator.generate(500)
#         self.segmentation_model.fit(data)
#         y_segmented = self.segmentation_model.predict(data)
#         datadf = pd.DataFrame({
#             'total_spent': [x.total_spent for x in data],
#             'total_purchases': [x.total_purchases for x in data],
#             'purchase_frequency': [x.purchase_frequency for x in data],
#             'will_buy_next_month': [x.will_buy_next_month for x in data],
#             'customer_segment': y_segmented
#         })
#         self.logistic_regression_model.fit(datadf)
#         accuracy, confusion = self.logistic_regression_model.evaluate()
#         return accuracy, confusion
#
# CustomerSegmentationExample().run()





# def GenerateData(n_customers):
#     total_spent = np.random.uniform(100, 5000, n_customers)
#     total_purchases = np.random.randint(1, 100, n_customers)
#     purchase_frequency = np.random.uniform(1, 30, n_customers)
#     will_buy_next_month = np.random.choice([0, 1], n_customers, p=[0.7, 0.3])
#     return pd.DataFrame({
#         'total_spent': total_spent,
#         'total_purchases': total_purchases,
#         'purchase_frequency': purchase_frequency,
#         'will_buy_next_month': will_buy_next_month
#     })
#
#
# def CustomerSegmentationModel(data):
#     # Aplicar K-Means con 3 clusters
#     X = data[['total_spent', 'total_purchases', 'purchase_frequency']]
#     scaler = StandardScaler()
#     X_sc = scaler.fit_transform(X)
#     y = data['will_buy_next_month']
#     model = KMeans(n_clusters=3, random_state=42, n_init=10)
#     data['customer_segment'] = model.fit_predict(X_sc, y)
#     return data
#
#
# def LogisticRegressionModel(data):
#     X = data[['total_spent', 'total_purchases', 'purchase_frequency', 'customer_segment']]
#     y = data['will_buy_next_month']
#     # Entrenar el modelo de regresión logística
#     model = LogisticRegression(max_iter=500)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     return model.fit(X_train, y_train)
#
# def predict(model, data):
#     data_segmented = CustomerSegmentationModel(data)
#     X = data_segmented[['total_spent', 'total_purchases', 'purchase_frequency', 'customer_segment']]
#     scaler = StandardScaler()
#     X_sc = scaler.fit_transform(X)
#     y = data_segmented['will_buy_next_month']
#
#
#     return model.predict(X)
#
# def evaluate_model(model, data):
#     X = data[['total_spent', 'total_purchases', 'purchase_frequency', 'customer_segment']]
#     y = data['will_buy_next_month']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     confusion = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
#     print("Precisión del modelo:", accuracy)
#     print("Matriz de Confusión:\n", confusion)
#     return accuracy, confusion
#
#
# # Crear el dataset
# np.random.seed(42)  # Para reproducibilidad
# n_customers = 200  # Número de clientes
#
# data = GenerateData(n_customers)
#
# data_transformed = CustomerSegmentationModel(data)
#
# # Entrenar el modelo de regresión logística
# model = LogisticRegressionModel(data_transformed)
#
# # Evaluar el modelo
# accuracy, confusion = evaluate_model(model, data_transformed)

