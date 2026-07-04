# Ryden Ranker

ML service for ranking place recommendations in Ryden.

The service receives a user, city, selected categories, and candidate places. It returns the same places sorted by predicted relevance.

## Features

- synthetic dataset generation;
- feature building from user, place, and request context;
- popularity baseline;
- `scikit-learn` ranking model;
- offline metrics: `NDCG`, `Recall`, `MAP`;
- FastAPI endpoints for ranking;
- tests for data generation, features, model pipeline, and API.

## Stack

Python, FastAPI, Pandas, scikit-learn, pytest.

## Setup

```powershell
cd "D:\Code\Python Projects\ryden-ranker"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Usage

Generate data:

```powershell
python -m ryden_ranker.data.generate --out data/processed
```

Train model:

```powershell
python -m ryden_ranker.train --data data/processed --model-out models/ranker.pkl
```

Evaluate model:

```powershell
python -m ryden_ranker.evaluate --data data/processed --model models/ranker.pkl
```

## Metrics

The project uses ranking metrics:

- `NDCG@5` shows how good the top recommendations are;
- `Recall@10` shows how many relevant places are found in the first page;
- `MAP@10` shows the average quality of the ranked top results.

The dataset is synthetic, so metric values are used as a pipeline sanity check, not as a claim about real product quality.

Run API:

```powershell
uvicorn ryden_ranker.api.app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## API

### `GET /health`

Returns service status and model information.

### `POST /rank`

Ranks candidate places.

Request example:

```json
{
  "user_id": "user_1",
  "city": "Novosibirsk",
  "category_ids": ["coffee"],
  "candidate_places": [
    {
      "place_id": "place_1",
      "title": "Coffee Spot",
      "description": "Small coffee place",
      "city": "Novosibirsk",
      "latitude": 55.03,
      "longitude": 82.92,
      "category_ids": ["coffee"],
      "created_at": "2026-06-01T00:00:00"
    }
  ],
  "recent_events": []
}
```

Response example:

```json
{
  "items": [
    {
      "place_id": "place_1",
      "score": 0.8123,
      "rank": 1,
      "reasons": ["matches requested or recent categories", "same city context"]
    }
  ],
  "model_version": "sklearn-logreg-v1"
}
```

### `POST /explain`

Returns ranking results with short reason strings for each place.

## Tests

```powershell
pytest
```
