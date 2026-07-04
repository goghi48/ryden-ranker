from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CandidatePlace(BaseModel):
    place_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = ""
    city: str = Field(..., min_length=1)
    latitude: float | None = None
    longitude: float | None = None
    category_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class RecentEvent(BaseModel):
    event_type: Literal[
        "place.viewed",
        "place.searched",
        "place.recommended",
        "place.clicked",
        "place.saved",
    ]
    place_id: str | None = None
    city: str | None = None
    category_ids: list[str] = Field(default_factory=list)
    event_time: datetime | None = None


class RankRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    category_ids: list[str] = Field(default_factory=list)
    candidate_places: list[CandidatePlace] = Field(..., min_length=1)
    recent_events: list[RecentEvent] = Field(default_factory=list)

    @field_validator("candidate_places")
    @classmethod
    def unique_candidates(cls, value: list[CandidatePlace]) -> list[CandidatePlace]:
        ids = [item.place_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate place ids must be unique")
        return value


class RankedItem(BaseModel):
    place_id: str
    score: float
    rank: int
    reasons: list[str] = Field(default_factory=list)


class RankResponse(BaseModel):
    items: list[RankedItem]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
