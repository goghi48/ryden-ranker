from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ryden_ranker.schemas import CandidatePlace, RecentEvent


FEATURE_COLUMNS = [
    "city_match",
    "category_overlap",
    "candidate_category_count",
    "user_history_count",
    "user_positive_history_count",
    "place_popularity",
    "place_positive_rate",
    "place_quality",
    "days_since_created",
    "distance_from_city_center",
]


CITY_CENTERS = {
    "Novosibirsk": (55.0302, 82.9204),
    "Moscow": (55.7558, 37.6173),
    "Saint Petersburg": (59.9311, 30.3609),
    "Kazan": (55.7961, 49.1064),
    "Yekaterinburg": (56.8389, 60.6057),
}


@dataclass(frozen=True)
class Dataset:
    users: pd.DataFrame
    places: pd.DataFrame
    events: pd.DataFrame
    train: pd.DataFrame
    valid: pd.DataFrame


def load_dataset(data_dir: Path) -> Dataset:
    return Dataset(
        users=pd.read_parquet(data_dir / "users.parquet"),
        places=pd.read_parquet(data_dir / "places.parquet"),
        events=pd.read_parquet(data_dir / "events.parquet"),
        train=pd.read_parquet(data_dir / "train.parquet"),
        valid=pd.read_parquet(data_dir / "valid.parquet"),
    )


def build_training_frame(interactions: pd.DataFrame, users: pd.DataFrame, places: pd.DataFrame) -> pd.DataFrame:
    frame = interactions.merge(users, on="user_id", how="left").merge(places, on="place_id", how="left")
    return _add_features(frame)


def build_candidate_frame(
    user_id: str,
    city: str,
    request_category_ids: list[str],
    candidates: list[CandidatePlace],
    recent_events: list[RecentEvent],
    historical_events: pd.DataFrame | None = None,
    places: pd.DataFrame | None = None,
) -> pd.DataFrame:
    candidate_frame = pd.DataFrame([_candidate_to_row(candidate) for candidate in candidates])
    if candidate_frame.empty:
        return pd.DataFrame(columns=["user_id", "place_id", *FEATURE_COLUMNS])

    history = _history_from_recent_events(recent_events)
    if historical_events is not None and not historical_events.empty:
        historical = historical_events.loc[historical_events["user_id"] == user_id].copy()
        history = pd.concat([history, historical], ignore_index=True)

    user_preferred = _preferred_categories_from_history(history, request_category_ids)
    popularity = _place_popularity(history)

    if places is not None and not places.empty:
        known_quality = places[["place_id", "quality"]].drop_duplicates("place_id")
        candidate_frame = candidate_frame.merge(known_quality, on="place_id", how="left")
    if "quality" not in candidate_frame:
        candidate_frame["quality"] = np.nan

    candidate_frame["user_id"] = user_id
    candidate_frame["home_city"] = city
    candidate_frame["preferred_categories"] = "|".join(user_preferred)
    candidate_frame["label"] = 0
    candidate_frame["place_popularity"] = candidate_frame["place_id"].map(popularity).fillna(0.0)
    candidate_frame["place_positive_rate"] = candidate_frame["place_id"].map(popularity).fillna(0.0)
    candidate_frame["user_history_count"] = len(history)
    candidate_frame["user_positive_history_count"] = int(history["event_type"].isin(["place.clicked", "place.saved"]).sum()) if not history.empty else 0
    return _add_features(candidate_frame, has_precomputed_history=True)


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURE_COLUMNS].fillna(0.0).astype(float)


def _add_features(frame: pd.DataFrame, has_precomputed_history: bool = False) -> pd.DataFrame:
    result = frame.copy()
    if "city" not in result:
        result["city"] = result["city_y"] if "city_y" in result else result.get("city_x", "")
    result["preferred_categories"] = result["preferred_categories"].fillna("")
    if "category_ids_y" not in result:
        result["category_ids_y"] = result["category_ids"] if "category_ids" in result else ""
    result["category_ids_y"] = result["category_ids_y"].fillna("")
    result["city_match"] = (result["home_city"].fillna(result["city"]) == result["city"]).astype(int)
    result["category_overlap"] = [
        len(_split_categories(user_categories) & _split_categories(place_categories))
        for user_categories, place_categories in zip(
            result["preferred_categories"],
            result["category_ids_y"],
            strict=False,
        )
    ]
    result["candidate_category_count"] = [
        len(_split_categories(categories)) for categories in result["category_ids_y"]
    ]
    if not has_precomputed_history:
        result["user_history_count"] = result.groupby("user_id")["place_id"].transform("count")
        result["user_positive_history_count"] = result.groupby("user_id")["label"].transform("sum")
        popularity = result.groupby("place_id")["label"].agg(["count", "mean"]).rename(
            columns={"count": "place_popularity", "mean": "place_positive_rate"}
        )
        result = result.merge(popularity, on="place_id", how="left", suffixes=("", "_computed"))
    if "quality" not in result:
        result["quality"] = 0.5
    result["place_quality"] = result["quality"].fillna(0.5)
    if "created_at" not in result:
        result["created_at"] = pd.Timestamp("2026-07-01")
    result["created_at"] = pd.to_datetime(result["created_at"]).fillna(pd.Timestamp("2026-07-01"))
    result["days_since_created"] = (
        pd.Timestamp("2026-07-01") - result["created_at"]
    ).dt.days.clip(lower=0)
    result["distance_from_city_center"] = [
        _rough_distance(city, lat, lon)
        for city, lat, lon in zip(result["city"], result["latitude"], result["longitude"], strict=False)
    ]
    for column in FEATURE_COLUMNS:
        if column not in result:
            result[column] = 0.0
    return result


def _candidate_to_row(candidate: CandidatePlace) -> dict[str, object]:
    return {
        "place_id": candidate.place_id,
        "title": candidate.title,
        "description": candidate.description,
        "city": candidate.city,
        "latitude": candidate.latitude,
        "longitude": candidate.longitude,
        "category_ids": "|".join(candidate.category_ids),
        "created_at": candidate.created_at,
    }


def _history_from_recent_events(events: list[RecentEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_type": event.event_type,
                "place_id": event.place_id,
                "city": event.city,
                "category_ids": "|".join(event.category_ids),
                "event_time": event.event_time,
            }
            for event in events
        ]
    )


def _preferred_categories_from_history(history: pd.DataFrame, fallback: list[str]) -> list[str]:
    counts: dict[str, float] = {}
    if not history.empty and "category_ids" in history:
        for row in history.itertuples(index=False):
            weight = 2.0 if getattr(row, "event_type", "") in {"place.clicked", "place.saved"} else 1.0
            for category in _split_categories(getattr(row, "category_ids", "")):
                counts[category] = counts.get(category, 0.0) + weight
    for category in fallback:
        counts[category] = counts.get(category, 0.0) + 1.0
    return [category for category, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]]


def _place_popularity(history: pd.DataFrame) -> dict[str, float]:
    if history.empty or "place_id" not in history:
        return {}
    return history.dropna(subset=["place_id"]).groupby("place_id").size().astype(float).to_dict()


def _split_categories(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, list):
        return {str(item) for item in value if str(item)}
    return {item for item in str(value).split("|") if item}


def _rough_distance(city: str, latitude: float | None, longitude: float | None) -> float:
    if latitude is None or longitude is None or pd.isna(latitude) or pd.isna(longitude):
        return 0.0
    center = CITY_CENTERS.get(city)
    if center is None:
        return 0.0
    lat0, lon0 = center
    return float(np.sqrt((latitude - lat0) ** 2 + (longitude - lon0) ** 2))
