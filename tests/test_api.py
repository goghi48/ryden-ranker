from fastapi.testclient import TestClient

from ryden_ranker.api.app import app


client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rank_validates_empty_candidates() -> None:
    response = client.post(
        "/rank",
        json={
            "user_id": "user_1",
            "city": "Novosibirsk",
            "category_ids": [],
            "candidate_places": [],
        },
    )

    assert response.status_code == 422


def test_rank_returns_sorted_items() -> None:
    response = client.post(
        "/rank",
        json={
            "user_id": "user_1",
            "city": "Novosibirsk",
            "category_ids": ["coffee"],
            "candidate_places": [
                {
                    "place_id": "a",
                    "title": "Coffee A",
                    "description": "coffee",
                    "city": "Novosibirsk",
                    "latitude": 55.03,
                    "longitude": 82.92,
                    "category_ids": ["coffee"],
                    "created_at": "2026-06-01T00:00:00",
                },
                {
                    "place_id": "b",
                    "title": "Park B",
                    "description": "park",
                    "city": "Novosibirsk",
                    "latitude": 55.04,
                    "longitude": 82.93,
                    "category_ids": ["parks"],
                    "created_at": "2026-06-02T00:00:00",
                },
            ],
            "recent_events": [],
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["items"][0]["rank"] == 1
    assert {item["place_id"] for item in payload["items"]} == {"a", "b"}
    assert "model_version" in payload
