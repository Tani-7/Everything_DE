FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: --ignore-requires-python -r requirements.txt || true


COPY src/ ./src
COPY models/ ./models
COPY .env .
COPY schema.sql .


CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
