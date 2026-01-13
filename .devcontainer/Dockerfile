FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
COPY .devcontainer/default.json /root/.cache/pydrive2fs/default.json
RUN dvc config core.no_scm true
RUN dvc pull