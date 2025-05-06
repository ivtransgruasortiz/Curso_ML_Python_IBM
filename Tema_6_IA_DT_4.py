import numpy as np
from scipy.constants import calorie

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class Snack:
    def __init__(self, calories, sugar, protein, fat, fiber, is_healthy=None):
        self.calories = calories
        self.sugar = sugar
        self.protein = protein
        self.fat = fat
        self.fiber = fiber
        self.is_healthy = is_healthy

    def to_vector(self):
        return [self.calories, self.sugar, self.protein, self.fat, self.fiber]


class SnackGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        snacks = []
        for _ in range(self.num_samples):
            calories = np.random.randint(50, 501)
            sugar = np.round(np.random.uniform(0, 50), 2)
            protein = np.round(np.random.uniform(0, 30), 2)
            fat = np.round(np.random.uniform(0, 30), 2)
            fiber = np.round(np.random.uniform(0, 15), 2)
            is_healthy = int((sugar <  15) &
                             ((protein >  5) | (fiber > 5)) &
                             (fat <  10) &
                             (calories <  200))
            snacks.append(Snack(calories, sugar, protein, fat, fiber, is_healthy))
        return snacks

class SnackClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, snacks):
        X = [snack.to_vector() for snack in snacks]
        y = [snack.is_healthy for snack in snacks]
        self.model.fit(X, y)

    def predict(self, snack):
        return self.model.predict([snack.to_vector()])[0]

class SnackRecommendationExample:
    def run(self):
        generator = SnackGenerator(200)
        training_data = generator.generate()
        classifier = SnackClassifier()
        classifier.fit(training_data)
        new_snack = Snack(150, 10, 6, 5, 3)
        prediction = classifier.predict(new_snack)

        # Mostrar la información y la predicción
        print("🔍 Snack Info:")
        print(
            f"Calories: {new_snack.calories}, Sugar: {new_snack.sugar}g, Protein: {new_snack.protein}g, "
            f"Fat: {new_snack.fat}g, Fiber: {new_snack.fiber}g")
        print("✅ Predicción: Este snack", "es saludable." if prediction == 1 else "no es saludable.")

# Ejecutar ejemplo
example = SnackRecommendationExample()
example.run()