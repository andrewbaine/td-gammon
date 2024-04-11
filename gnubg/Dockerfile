# syntax=docker/dockerfile:1

FROM ubuntu:jammy AS build

RUN <<HERE
apt update
apt install -y build-essential flex bison libglib2.0-dev libreadline-dev
HERE

COPY gnubg-release-1.08.002-sources.tar.gz .
RUN tar -xzvf gnubg-release-1.08.002-sources.tar.gz
WORKDIR /gnubg-1.08.002

RUN ./configure
RUN make
RUN make install

ENTRYPOINT ["/usr/local/bin/gnubg", "-q", "-t"]
CMD []
