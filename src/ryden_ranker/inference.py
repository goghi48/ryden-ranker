from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ryden_ranker.features.build import build_candidate_frame
from ryden_ranker.models.ranker import load_ranker_or_baseline
from ryden_ranker.schemas import CandidatePlace, RankedItem, RecentEvent


@dataclass
class RankingService:
    model_path: Path
    historical_events: pd.DataFrame | None = None
    places: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.model = load_ranker_or_baseline(self.model_path)

    @property
    def model_version(self) -> str:
        return getattr(self.model, "model_version", "unknown")

    @property
    def model_loaded(self) -> bool:
        return self.model_path.exists()

    def rank(
        self,
        user_id: str,
        city: str,
        category_ids: list[str],
        candidate_places: list[CandidatePlace],
        recent_events: list[RecentEvent],
    ) -> list[RankedItem]:
        frame = build_candidate_frame(
            user_id=user_id,
            city=city,
            request_category_ids=category_ids,
            candidates=candidate_places,
            recent_events=recent_events,
            historical_events=self.historical_events,
            places=self.places,
        )
        scores = self.model.predict(frame)
        ranked = frame[["place_id", "category_overlap", "city_match", "place_popularity"]].copy()
        ranked["score"] = scores
        ranked = ranked.sort_values(["score", "place_id"], ascending=[False, True]).reset_index(drop=True)

        items = []
        for index, row in ranked.iterrows():
            items.append(
                RankedItem(
                    place_id=str(row["place_id"]),
                    score=round(float(row["score"]), 6),
                    rank=index + 1,
                    reasons=_reasons(row),
                )
            )
        return items


def _reasons(row: pd.Series) -> list[str]:
    reasons = []
    if float(row.get("category_overlap", 0)) > 0:
        reasons.append("matches requested or recent categories")
    if float(row.get("city_match", 0)) > 0:
        reasons.append("same city context")
    if float(row.get("place_popularity", 0)) > 0:
        reasons.append("has positive user history")
    if not reasons:
        reasons.append("cold-start fallback score")
    return reasons[:3]
