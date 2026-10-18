# The pipeline in a container (spec 12 deliverable 1). Data lives in a mounted volume:
#   docker run --rm -v ~/cryptoval-data:/data cryptoval              -> make all
#   docker run --rm -p 8501:8501 -v ~/cryptoval-data:/data cryptoval make dashboard
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends make git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU torch first, exactly as CI installs it: the default index would pull the CUDA build.
RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .
RUN .venv/bin/pip install --no-cache-dir -e . \
    && cd dbt && ../.venv/bin/dbt deps --profiles-dir .

ENV CRYPTOVAL_DATA_ROOT=/data
VOLUME /data
EXPOSE 8501
# Streamlit must listen beyond localhost inside a container.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 STREAMLIT_SERVER_HEADLESS=true
CMD ["make", "all"]
