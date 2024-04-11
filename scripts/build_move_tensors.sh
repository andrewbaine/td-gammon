#!/bin/bash
set -e
set -x

D=$(date +%s)

mkdir -p var/move_tensors

docker run \
       --mount type=bind,src=$(pwd)/var/move_tensors,target=/var/move_tensors \
       td-gammon move-tensors --prefix /var/move_tensors/tensors-${D}

pushd var/move_tensors
pwd
rm -f current
ln -sf tensors-${D} current
popd
