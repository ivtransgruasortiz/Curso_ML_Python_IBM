from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class Project:
    def __init__(self, name, team_size, budget, duration_months, realtime_required, needs_offline, target_users,
                 recommended_platform=None):
        self.name = name
        self.team_size = team_size
        self.budget = budget
        self.duration_months = duration_months
        self.realtime_required = int(realtime_required)  # True/False → 1/0
        self.needs_offline = int(needs_offline)
        self.target_users = target_users
        self.recommended_platform = recommended_platform

    def to_features(self, label_encoder=None):
        target_users_encoded = self.target_users
        if label_encoder:
            target_users_encoded = label_encoder.transform([self.target_users])[0]
        return [
            self.team_size,
            self.budget,
            self.duration_months,
            self.realtime_required,
            self.needs_offline,
            target_users_encoded
        ]


class ProjectDataset:
    def __init__(self, projects):
        self.projects = projects
        self.label_encoder_users = LabelEncoder()
        self.label_encoder_platform = LabelEncoder()
        self._fit_encoders()

    def _fit_encoders(self):
        user_types = [p.target_users for p in self.projects]
        platform_labels = [p.recommended_platform for p in self.projects if p.recommended_platform]
        self.label_encoder_users.fit(user_types)
        self.label_encoder_platform.fit(platform_labels)

    def get_X_y(self):
        X = []
        y = []
        for p in self.projects:
            if p.recommended_platform:
                X.append(p.to_features(self.label_encoder_users))
                y.append(p.recommended_platform)
        y_encoded = self.label_encoder_platform.transform(y)
        return X, y_encoded

    def decode_platform(self, label):
        return self.label_encoder_platform.inverse_transform([label])[0]


class PlatformRecommender:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.dataset = None

    def train(self, dataset: ProjectDataset):
        self.dataset = dataset
        X, y = dataset.get_X_y()
        self.model.fit(X, y)

    def evaluate(self):
        """Evalúa el modelo con datos de prueba y retorna la precisión."""
        X, y = self.dataset.get_X_y()
        y_pred = self.model.predict(X)
        return accuracy_score(y, y_pred)

    def predict(self, project: Project):
        features = project.to_features(self.dataset.label_encoder_users)
        label = self.model.predict([features])[0]
        return self.dataset.decode_platform(label)

projects = [
    Project("AppGlobal", 5, 25.0, 6, True, False, "global", "web"),
    Project("IntranetCorp", 10, 40.0, 12, False, True, "empresa", "desktop"),
    Project("LocalDelivery", 3, 20.0, 4, True, True, "local", "mobile"),
    Project("CloudDashboard", 6, 50.0, 8, True, False, "empresa", "web"),
    Project("OfflineTool", 4, 15.0, 6, False, True, "local", "desktop"),
    Project("SocialBuzz", 2, 10.0, 3, True, False, "global", "mobile"),
]

new_project = Project("AIChatApp", 4, 30.0, 5, True, False, "global")

dataset = ProjectDataset(projects)
recommender = PlatformRecommender()
recommender.train(dataset)

prediction = recommender.predict(new_project)
print(f"Plataforma recomendada: {prediction}")
