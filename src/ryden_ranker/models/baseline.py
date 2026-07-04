from __future__ import annotations

import pandas as pd


class PopularityBaseline:
    """Simple non-ML ranker used for cold start and model comparison."""

    model_version = "baseline-popularity-v1"

    def __init__(self, popularity: dict[str, float] | None = None) -> None:
        self.popularity = popularity or {}

    @classmethod
    def fit(cls, train_frame: pd.DataFrame) -> "PopularityBaseline":
        popularity = train_frame.groupby("place_id")["label"].mean().astype(float).to_dict()
        return cls(popularity)

    def predict(self, frame: pd.DataFrame) -> list[float]:
        scores = []
        for row in frame.itertuples(index=False):
            score = float(self.popularity.get(row.place_id, 0.0))
            score += 0.15 * float(getattr(row, "category_overlap", 0.0))
            score += 0.05 * float(getattr(row, "city_match", 0.0))
            score += 0.01 * float(getattr(row, "place_quality", 0.5))
            scores.append(score)
        return scores
