#!/bin/bash
set -e
set -x

D=$(date +%s)

DIR=var/move_tensors
mkdir -p $DIR
F=tensors-${D}

docker run \
       --mount type=bind,src=$(pwd)/${DIR},target=/${DIR} \
       td-gammon move-tensors \
       --dir /${DIR}/${F}

pushd ${DIR}
pwd
rm -f current
ln -sf ${F} current
popd
