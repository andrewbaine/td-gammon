#!/bin/bash

set -e
set -x

GAMES="100"
while getopts ":g:m:t:" opt; do
    case $opt in
        g)
            GAMES="${OPTARG}"
            ;;
        *)
            echo "bad command"
            exit 1
    esac
done

shift "$((OPTIND-1))"

if [ -z "$GAMES" ]
then
    echo "set GAMES variable"
    exit 1
fi

MODEL=$1
if [ -z "$MODEL" ]
then
    echo "set MODEL variable"
    exit 1
fi
if docker run --rm --gpus all hello-world >/dev/null 2>/dev/null; then
    GPU_ARGS="--gpus all";
else
    GPU_ARGS="";
fi

WD=$(pwd)

LOGS_DIR=${WD}/var/eval-logs
mkdir -p ${LOGS_DIR}

GNUBG_ERR=${LOGS_DIR}/gnubg.err
GNUBG_OUT=${LOGS_DIR}/gnubg.out
PY_ERR=${LOGS_DIR}/py.err
PY_OUT=${LOGS_DIR}/py.out

rm -f $GNUBG_ERR $GNUBG_OUT $PY_ERR $PY_OUT
touch $GNUBG_ERR $GNUBG_OUT $PY_ERR $PY_OUT

COMMAND_1="evaluate --load-model /var/model/$(basename ${MODEL}) --games $GAMES"
docker run --rm ${GPU_ARGS} \
       --mount type=bind,src=${WD}/$(dirname ${MODEL}),target=/var/model \
       -i td-gammon $COMMAND_1 \
       >${PY_OUT} \
       2>$PY_ERR \
       < <(tail -f ${GNUBG_OUT}) &
PYTHON_JOB="$!"

docker run --rm \
       -i gnubg \
       >${GNUBG_OUT} \
       2>${GNUBG_ERR=} \
       < <(tail -f ${PY_OUT}) &
GNUBG_JOB="$!"

wait $PYTHON_JOB

A=$(grep "gnubg wins a backgammon" $GNUBG_OUT | wc -l)
B=$(grep "gnubg wins a gammon" $GNUBG_OUT | wc -l)
C=$(grep "gnubg wins a single game" $GNUBG_OUT | wc -l)

D=$(grep " wins a single game" $GNUBG_OUT | grep -v gnubg | wc -l)
E=$(grep " wins a gammon" $GNUBG_OUT | grep -v gnubg | wc -l)
F=$(grep " wins a backgammon" $GNUBG_OUT | grep -v gnubg | wc -l)

CALCULATION="(-3 * $A - 2 * $B - $C + $D + 2 * $E + 3 * $F) / ($A + $B + $C + $D + $E + $F)"
PPG=$(echo "scale=2; $CALCULATION" | bc)

echo "$PPG $A $B $C $D $E $F"
