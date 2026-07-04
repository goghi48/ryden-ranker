from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text


@dataclass(frozen=True)
class ClickHouseConfig:
    dsn: str


class ClickHouseEventsAdapter:
    """Read Ryden analytics events when the production ClickHouse is available."""

    def __init__(self, config: ClickHouseConfig) -> None:
        self.engine = create_engine(config.dsn)

    def load_events(self, days: int = 30) -> pd.DataFrame:
        query = text(
            """
            SELECT
                event_id,
                event_type,
                event_time,
                user_id,
                session_id,
                place_id,
                city,
                arrayStringConcat(category_ids, '|') AS category_ids
            FROM analytics_events
            WHERE event_time >= now() - INTERVAL :days DAY
              AND event_type IN (
                'place.viewed',
                'place.searched',
                'place.recommended',
                'place.clicked',
                'place.saved'
              )
            """
        )
        return pd.read_sql_query(query, self.engine, params={"days": days})
