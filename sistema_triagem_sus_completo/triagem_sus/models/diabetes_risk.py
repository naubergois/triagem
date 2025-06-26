import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DiabetesRiskModel:
    """Modelo simples para previsão de risco de diabetes."""

    model: Optional[LogisticRegression] = None
    scaler: Optional[StandardScaler] = None

    def train(self) -> float:
        """Treina o modelo usando o dataset de exemplo do scikit-learn."""
        data = load_diabetes()
        X = data.data[:, [0, 3]]  # age e blood pressure
        y = (data.target > np.median(data.target)).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train_scaled, y_train)

        acc = self.model.score(X_test_scaled, y_test)
        return acc

    def save(self, path: str) -> None:
        if self.model is None or self.scaler is None:
            raise ValueError("Modelo não treinado")
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]

    def predict_proba(self, age: float, systolic_bp: float) -> float:
        if self.model is None or self.scaler is None:
            raise ValueError("Modelo não carregado")
        features = np.array([[age, systolic_bp]])
        features_scaled = self.scaler.transform(features)
        prob = self.model.predict_proba(features_scaled)[0, 1]
        return float(prob)


def age_range_to_value(age_range: str) -> float:
    mapping = {
        "0-15": 7.5,
        "16-30": 23,
        "31-45": 38,
        "46-60": 53,
        "61-75": 68,
        "76+": 80,
    }
    return mapping.get(age_range, 40)


if __name__ == "__main__":
    model = DiabetesRiskModel()
    acc = model.train()
    path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
    model.save(path)
    print(f"Modelo treinado com acurácia {acc:.2f} e salvo em {path}")

