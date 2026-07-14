# Build context is the repository root (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir \
        numpy pandas pyarrow duckdb "dbt-duckdb>=1.7" kafka-python requests

# The heliostream package is reused for the simulator / OMNI / NOAA sources and
# the model. Install it WITHOUT its heavy deps (torch is only needed for the
# separate 'train' step, not for the ingestion/transform services).
COPY heliostream /app/heliostream
RUN pip install --no-cache-dir --no-deps -e /app/heliostream

COPY data_engineering /app/data_engineering
ENV PYTHONPATH=/app/data_engineering
WORKDIR /app/data_engineering
