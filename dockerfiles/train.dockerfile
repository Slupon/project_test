# Base image

# Hay que moverlo de lugar a dockerfiles y substituir el que ya hay !!!
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN mkdir -p models reports/figures

ENTRYPOINT ["python", "-u", "src/project_test/train.py"]