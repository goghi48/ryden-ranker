from pathlib import Path

from ryden_ranker.data.generate import GenerateConfig, generate_dataset, write_dataset
from ryden_ranker.evaluate import evaluate
from ryden_ranker.train import train


def test_training_produces_model_and_metrics(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    model_path = tmp_path / "ranker.pkl"
    frames = generate_dataset(GenerateConfig(users=35, places=120, impressions_per_user=16, seed=21))
    write_dataset(frames, data_dir)

    metrics = train(data_dir, model_path)
    evaluated = evaluate(data_dir, model_path)

    assert model_path.exists()
    assert {"model_ndcg@5", "model_map@10", "baseline_ndcg@5"}.issubset(metrics)
    assert {"model_ndcg@5", "model_map@10", "baseline_ndcg@5"}.issubset(evaluated)
    assert metrics["model_ndcg@5"] >= metrics["baseline_ndcg@5"]
