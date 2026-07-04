from __future__ import annotations

import argparse
from pathlib import Path

from ryden_ranker.features.build import build_training_frame, load_dataset
from ryden_ranker.metrics import ranking_metrics
from ryden_ranker.models.baseline import PopularityBaseline
from ryden_ranker.models.ranker import SklearnPlaceRanker


def evaluate(data_dir: Path, model_path: Path) -> dict[str, float]:
    dataset = load_dataset(data_dir)
    train_frame = build_training_frame(dataset.train, dataset.users, dataset.places)
    valid_frame = build_training_frame(dataset.valid, dataset.users, dataset.places)

    baseline = PopularityBaseline.fit(train_frame)
    valid_frame["baseline_score"] = baseline.predict(valid_frame)

    ranker = SklearnPlaceRanker.load(model_path)
    valid_frame["score"] = ranker.predict(valid_frame)

    model_metrics = ranking_metrics(valid_frame, "score")
    baseline_metrics = ranking_metrics(valid_frame, "baseline_score")
    return {
        **{f"model_{key}": value for key, value in model_metrics.items()},
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Ryden ranking model.")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", type=Path, default=Path("models/ranker.pkl"))
    args = parser.parse_args()

    metrics = evaluate(args.data, args.model)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
