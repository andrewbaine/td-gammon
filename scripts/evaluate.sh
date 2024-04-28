#!/bin/bash

set -e
set -x

games="100"
while getopts ":d:g:" opt; do
    case $opt in
	d)
	    epc_db="${OPTARG}"
	    ;;
        g)
            games="${OPTARG}"
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

model=$1
if [ -z "$model" ]
then
    echo "set model variable"
    exit 1
fi

force_cuda=""
declare -a gpu_args
if docker run --rm --gpus all hello-world >/dev/null 2>/dev/null; then
    read -r -a gpu_args < <(echo "--gpus all")
    force_cuda="--force-cuda"
fi

declare -a epc_db_args
declare -a epc_docker_args
if [ -n "$epc_db" ]
then
    read -r -a epc_db_args < <(echo "--epc-db /var/epc_db")
    read -r -a epc_docker_args < <(echo "--mount type=bind,src=$(pwd)/${epc_db},target=/var/epc_db")
fi

WD=$(pwd)

LOGS_DIR=${WD}/var/eval-logs
mkdir -p ${LOGS_DIR}

GNUBG_ERR=$(mktemp ${LOGS_DIR}/gnubg.err.XXXXXX)
GNUBG_OUT=$(mktemp ${LOGS_DIR}/gnubg.out.XXXXXX)
PY_ERR=$(mktemp ${LOGS_DIR}/py.err.XXXXXX)
PY_OUT=$(mktemp ${LOGS_DIR}/py.out.XXXXXX)

touch $GNUBG_ERR $GNUBG_OUT $PY_ERR $PY_OUT

docker run --rm "${gpu_args[@]}" \
       --mount type=bind,src=${WD}/$(dirname ${model}),target=/var/model \
       "${epc_docker_args[@]}" \
       -i td-gammon evaluate ${force_cuda} --load-model "/var/model/$(basename ${model})" --games "${games}" "${epc_db_args[@]}" \
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
