from __future__ import annotations

import numpy as np
import pandas as pd


def ranking_metrics(frame: pd.DataFrame, score_column: str = "score", k_values: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    metrics: dict[str, float] = {}
    grouped = list(frame.groupby("user_id"))
    for k in k_values:
        metrics[f"ndcg@{k}"] = float(np.mean([_ndcg_at_k(group, score_column, k) for _, group in grouped]))
        metrics[f"recall@{k}"] = float(np.mean([_recall_at_k(group, score_column, k) for _, group in grouped]))
    metrics["map@10"] = float(np.mean([_average_precision_at_k(group, score_column, 10) for _, group in grouped]))
    return metrics


def _ndcg_at_k(group: pd.DataFrame, score_column: str, k: int) -> float:
    ranked = group.sort_values(score_column, ascending=False).head(k)
    labels = ranked["label"].to_numpy(dtype=float)
    dcg = np.sum((2**labels - 1) / np.log2(np.arange(2, len(labels) + 2)))
    ideal = np.sort(group["label"].to_numpy(dtype=float))[::-1][:k]
    idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _recall_at_k(group: pd.DataFrame, score_column: str, k: int) -> float:
    positives = float(group["label"].sum())
    if positives == 0:
        return 0.0
    found = group.sort_values(score_column, ascending=False).head(k)["label"].sum()
    return float(found / positives)


def _average_precision_at_k(group: pd.DataFrame, score_column: str, k: int) -> float:
    ranked = group.sort_values(score_column, ascending=False).head(k)
    positives = 0
    precision_sum = 0.0
    for index, label in enumerate(ranked["label"].to_numpy(dtype=int), start=1):
        if label:
            positives += 1
            precision_sum += positives / index
    total_positives = int(group["label"].sum())
    if total_positives == 0:
        return 0.0
    return float(precision_sum / min(total_positives, k))
