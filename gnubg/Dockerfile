# syntax=docker/dockerfile:1

FROM ubuntu:jammy AS base


RUN apt update && apt install -y build-essential flex bison libglib2.0-dev libreadline-dev && apt clean
COPY gnubg-release-1.08.002-sources.tar.gz .
RUN tar -xzvf gnubg-release-1.08.002-sources.tar.gz
WORKDIR /gnubg-1.08.002

RUN ./configure
RUN make
RUN make install

FROM base
COPY --from=0 /usr/local/bin/gnubg /bin/gnubg
CMD ["/bin/gnubg"]

