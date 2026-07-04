from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CITIES = ["Novosibirsk", "Moscow", "Saint Petersburg", "Kazan", "Yekaterinburg"]
CATEGORIES = [
    "coffee",
    "parks",
    "food",
    "culture",
    "work",
    "sport",
    "nightlife",
    "shopping",
]
EVENT_WEIGHTS = {
    "place.viewed": 0.15,
    "place.clicked": 0.65,
    "place.saved": 1.0,
}


@dataclass(frozen=True)
class GenerateConfig:
    users: int = 350
    places: int = 900
    impressions_per_user: int = 70
    seed: int = 48


def generate_dataset(config: GenerateConfig) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    users = _generate_users(rng, config.users)
    places = _generate_places(rng, config.places)
    interactions = _generate_interactions(rng, users, places, config.impressions_per_user)
    events = _interactions_to_events(rng, interactions)
    train, valid = _split_by_user(interactions)
    return {
        "users": users,
        "places": places,
        "events": events,
        "train": train,
        "valid": valid,
    }


def _generate_users(rng: np.random.Generator, n_users: int) -> pd.DataFrame:
    rows = []
    for idx in range(n_users):
        home_city = rng.choice(CITIES, p=[0.36, 0.24, 0.18, 0.1, 0.12])
        preferred = rng.choice(CATEGORIES, size=rng.integers(2, 5), replace=False)
        rows.append(
            {
                "user_id": f"user_{idx:04d}",
                "home_city": home_city,
                "preferred_categories": "|".join(preferred),
                "activity_level": float(rng.beta(2, 5)),
            }
        )
    return pd.DataFrame(rows)


def _generate_places(rng: np.random.Generator, n_places: int) -> pd.DataFrame:
    city_centers = {
        "Novosibirsk": (55.0302, 82.9204),
        "Moscow": (55.7558, 37.6173),
        "Saint Petersburg": (59.9311, 30.3609),
        "Kazan": (55.7961, 49.1064),
        "Yekaterinburg": (56.8389, 60.6057),
    }
    rows = []
    for idx in range(n_places):
        city = rng.choice(CITIES, p=[0.35, 0.22, 0.18, 0.12, 0.13])
        categories = rng.choice(CATEGORIES, size=rng.integers(1, 4), replace=False)
        lat_center, lon_center = city_centers[city]
        quality = float(rng.beta(4, 2))
        rows.append(
            {
                "place_id": f"place_{idx:05d}",
                "title": f"{city} {categories[0].title()} Spot {idx}",
                "description": f"A local {', '.join(categories)} place in {city}.",
                "city": city,
                "latitude": float(lat_center + rng.normal(0, 0.035)),
                "longitude": float(lon_center + rng.normal(0, 0.045)),
                "category_ids": "|".join(categories),
                "quality": quality,
                "created_at": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 180))),
            }
        )
    return pd.DataFrame(rows)


def _generate_interactions(
    rng: np.random.Generator,
    users: pd.DataFrame,
    places: pd.DataFrame,
    impressions_per_user: int,
) -> pd.DataFrame:
    rows = []
    places_by_city = {city: frame.reset_index(drop=True) for city, frame in places.groupby("city")}
    global_popularity = rng.gamma(shape=2.0, scale=1.0, size=len(places))
    popularity_by_place = dict(zip(places["place_id"], global_popularity, strict=True))

    for user in users.itertuples(index=False):
        user_categories = set(user.preferred_categories.split("|"))
        candidate_city = user.home_city
        city_places = places_by_city[candidate_city]
        sampled = city_places.sample(
            n=min(impressions_per_user, len(city_places)),
            replace=False,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        for place in sampled.itertuples(index=False):
            place_categories = set(place.category_ids.split("|"))
            category_overlap = len(user_categories & place_categories)
            recency_days = (pd.Timestamp("2026-07-01") - place.created_at).days
            city_match = int(place.city == user.home_city)
            raw_score = (
                -2.0
                + 1.35 * category_overlap
                + 1.15 * city_match
                + 1.9 * place.quality
                + 0.45 * user.activity_level
                + 0.12 * np.log1p(popularity_by_place[place.place_id])
                - 0.002 * recency_days
            )
            probability = 1 / (1 + np.exp(-raw_score))
            label = int(rng.random() < probability)
            event_type = rng.choice(
                ["place.saved", "place.clicked", "place.viewed"] if label else ["place.viewed", "place.recommended"],
                p=[0.25, 0.5, 0.25] if label else [0.7, 0.3],
            )
            rows.append(
                {
                    "user_id": user.user_id,
                    "place_id": place.place_id,
                    "city": candidate_city,
                    "category_ids": user.preferred_categories,
                    "label": label,
                    "event_type": event_type,
                    "event_time": pd.Timestamp("2026-06-01")
                    + pd.Timedelta(days=int(rng.integers(0, 30))),
                }
            )
    return pd.DataFrame(rows)


def _interactions_to_events(rng: np.random.Generator, interactions: pd.DataFrame) -> pd.DataFrame:
    events = interactions[
        ["user_id", "place_id", "city", "category_ids", "event_type", "event_time"]
    ].copy()
    events["event_id"] = [f"event_{idx:07d}" for idx in range(len(events))]
    events["session_id"] = [f"session_{rng.integers(0, 20_000):05d}" for _ in range(len(events))]
    return events[
        ["event_id", "event_type", "event_time", "user_id", "session_id", "place_id", "city", "category_ids"]
    ]


def _split_by_user(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_events = interactions.sort_values(["user_id", "event_time", "place_id"]).copy()
    sorted_events["row_number"] = sorted_events.groupby("user_id").cumcount()
    sorted_events["row_count"] = sorted_events.groupby("user_id")["place_id"].transform("count")
    valid_start = np.maximum((sorted_events["row_count"] * 0.8).astype(int), sorted_events["row_count"] - 1)
    valid_mask = sorted_events["row_number"] >= valid_start
    train = sorted_events.loc[~valid_mask].drop(columns=["row_number", "row_count"])
    valid = sorted_events.loc[valid_mask].drop(columns=["row_number", "row_count"])
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def write_dataset(frames: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Ryden ranking data.")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--users", type=int, default=350)
    parser.add_argument("--places", type=int, default=900)
    parser.add_argument("--impressions-per-user", type=int, default=70)
    parser.add_argument("--seed", type=int, default=48)
    args = parser.parse_args()

    config = GenerateConfig(
        users=args.users,
        places=args.places,
        impressions_per_user=args.impressions_per_user,
        seed=args.seed,
    )
    write_dataset(generate_dataset(config), args.out)
    print(f"Wrote synthetic Ryden ranking dataset to {args.out}")


if __name__ == "__main__":
    main()
