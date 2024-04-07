# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 as base

RUN <<HERE
apt update
apt-get install -y python3 python3-pip
apt-get install -y build-essential flex bison libglib2.0-dev libreadline-dev
apt-get clean
HERE

COPY gnubg-release-1.08.002-sources.tar.gz .

RUN <<HERE
tar -xzvf gnubg-release-1.08.002-sources.tar.gz
cd gnubg-1.08.002
./configure
make
make install
cd ..
rm -rf gnubg-1.08.002
HERE


WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY src src

ENTRYPOINT ["python3", "/app/src/main.py"]
