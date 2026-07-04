from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI

from ryden_ranker import __version__
from ryden_ranker.inference import RankingService
from ryden_ranker.schemas import HealthResponse, RankRequest, RankResponse


MODEL_PATH = Path(os.getenv("RYDEN_RANKER_MODEL_PATH", "models/ranker.pkl"))

app = FastAPI(
    title="Ryden Ranker",
    version=__version__,
    description="ML ranking service for Ryden place recommendations.",
)


@lru_cache(maxsize=1)
def get_ranking_service() -> RankingService:
    return RankingService(model_path=MODEL_PATH)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_ranking_service()
    return HealthResponse(
        status="ok",
        model_loaded=service.model_loaded,
        model_version=service.model_version,
    )


@app.post("/rank", response_model=RankResponse)
def rank(request: RankRequest) -> RankResponse:
    service = get_ranking_service()
    items = service.rank(
        user_id=request.user_id,
        city=request.city,
        category_ids=request.category_ids,
        candidate_places=request.candidate_places,
        recent_events=request.recent_events,
    )
    return RankResponse(items=items, model_version=service.model_version)


@app.post("/explain", response_model=RankResponse)
def explain(request: RankRequest) -> RankResponse:
    # v1 returns the same ranked payload with per-item reason strings.
    return rank(request)
