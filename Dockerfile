FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RYDEN_RANKER_MODEL_PATH=/app/models/ranker.pkl

COPY pyproject.toml README.md ./
COPY src ./src
COPY models/.gitkeep ./models/.gitkeep

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "ryden_ranker.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
