# The software of the analyses (CUDA 12.6, Python 3.10, requirements.txt). The code, data and outputs stay in this
# folder, mounted at /work.
# Build:  docker build -t e2m-analyses .
# Run:    docker run --gpus all --rm -v "<this folder>":/work e2m-analyses python run_all.py
FROM nvidia/cuda:12.6.3-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    E2M_COMMIT=b201b223154aaa15518ec8faab9df01bde04bec5 \
    PATH=/opt/venv/bin:$PATH

# python3.10-dev and build-essential: inmoose is built from source
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3.10-venv python3.10-dev build-essential libgomp1 git \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv \
    && python -m pip install pip==26.1.2

COPY requirements.txt /opt/
RUN python -m pip install -r /opt/requirements.txt

WORKDIR /work
