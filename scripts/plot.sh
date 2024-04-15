#!/bin/bash

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
    echo "set GAMES variable" >&2
    exit 1
fi

DIR=$1

if [ -z "${DIR}" ]
then
   echo "pass DIR" >&2
   exit 1
fi

for x in $DIR/model.*.pt
do
    data=$(./scripts/evaluate.sh -g $GAMES $x)
    printf "%s\t%s\n" "$(basename ${x})" "${data}" | tee -a ${DIR}/plot-${GAMES}.txt
done
