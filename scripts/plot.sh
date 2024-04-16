#!/bin/bash
set -e
set -x

while getopts ":g:" opt; do
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

PLOT_FILE=${DIR}/plot-${GAMES}.txt
touch ${PLOT_FILE}
chmod a+w ${PLOT_FILE}
for x in $DIR/model.*.pt
do
    SHORT_NAME=$(basename ${x})
    grep "$SHORT_NAME" $PLOT_FILE || \
        DATA=$(./scripts/evaluate.sh -g $GAMES $x) \
            printf "%s\t%s\n" "$SHORT_NAME" "${data}" \
            | tee -a $PLOT_FILE
done
chmod a-w ${PLOT_FILE}
