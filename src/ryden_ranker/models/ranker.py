from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ryden_ranker.features.build import FEATURE_COLUMNS, feature_matrix
from ryden_ranker.models.baseline import PopularityBaseline


class SklearnPlaceRanker:
    model_version = "sklearn-logreg-v1"

    def __init__(self) -> None:
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=500, random_state=48)),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> "SklearnPlaceRanker":
        self.model.fit(feature_matrix(frame), frame["label"].astype(int))
        return self

    def predict(self, frame: pd.DataFrame) -> list[float]:
        probabilities = self.model.predict_proba(feature_matrix(frame))[:, 1]
        return [float(value) for value in probabilities]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self.model, file)

    @classmethod
    def load(cls, path: Path) -> "SklearnPlaceRanker":
        instance = cls()
        with path.open("rb") as file:
            instance.model = pickle.load(file)
        return instance

    def feature_importance(self) -> list[tuple[str, float]]:
        classifier = self.model.named_steps["classifier"]
        values = abs(classifier.coef_[0])
        return sorted(zip(FEATURE_COLUMNS, values, strict=True), key=lambda item: item[1], reverse=True)


def load_ranker_or_baseline(model_path: Path) -> SklearnPlaceRanker | PopularityBaseline:
    if model_path.exists():
        try:
            return SklearnPlaceRanker.load(model_path)
        except Exception:
            return PopularityBaseline()
    return PopularityBaseline()
