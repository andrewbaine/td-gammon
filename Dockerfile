# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 as base

RUN <<HERE
apt update
apt-get install -y python3 python3-pip
apt-get clean
HERE

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY src src

ENTRYPOINT ["python3", "/app/src/main.py"]
CMD ["--help"]