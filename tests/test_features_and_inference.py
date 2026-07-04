from datetime import datetime
from pathlib import Path

from ryden_ranker.features.build import FEATURE_COLUMNS, build_candidate_frame
from ryden_ranker.inference import RankingService
from ryden_ranker.schemas import CandidatePlace, RecentEvent


def _candidates() -> list[CandidatePlace]:
    return [
        CandidatePlace(
            place_id="a",
            title="Coffee A",
            description="coffee",
            city="Novosibirsk",
            latitude=55.03,
            longitude=82.92,
            category_ids=["coffee"],
            created_at=datetime(2026, 6, 1),
        ),
        CandidatePlace(
            place_id="b",
            title="Park B",
            description="park",
            city="Novosibirsk",
            latitude=None,
            longitude=None,
            category_ids=["parks"],
            created_at=None,
        ),
    ]


def test_feature_builder_handles_empty_history_and_missing_coordinates() -> None:
    frame = build_candidate_frame(
        user_id="user_1",
        city="Novosibirsk",
        request_category_ids=["coffee"],
        candidates=_candidates(),
        recent_events=[],
    )

    assert list(frame["place_id"]) == ["a", "b"]
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert frame[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert frame.loc[frame["place_id"] == "a", "category_overlap"].item() == 1


def test_ranking_is_sorted_and_preserves_candidate_ids(tmp_path: Path) -> None:
    service = RankingService(model_path=tmp_path / "missing.cbm")

    ranked = service.rank(
        user_id="user_1",
        city="Novosibirsk",
        category_ids=["coffee"],
        candidate_places=_candidates(),
        recent_events=[
            RecentEvent(
                event_type="place.clicked",
                place_id="a",
                city="Novosibirsk",
                category_ids=["coffee"],
            )
        ],
    )

    assert {item.place_id for item in ranked} == {"a", "b"}
    assert [item.rank for item in ranked] == [1, 2]
    assert ranked[0].score >= ranked[1].score
