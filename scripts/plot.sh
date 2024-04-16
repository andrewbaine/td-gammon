#!/bin/bash
set -e
set -x

while getopts ":g:" opt; do
    case $opt in
        g)
            games="${OPTARG}"
            ;;
        *)
            echo "bad command"
            exit 1
    esac
done

shift "$((OPTIND-1))"

if [ -z "$games" ]
then
    echo "-g GAMES" >&2
    exit 1
fi

dir=$1

if [ -z "${dir}" ]
then
   echo "pass dir" >&2
   exit 1
fi

plot_file=${dir}/plot-${games}.txt
touch "${plot_file}"
chmod a+w "${plot_file}"
for x in "${dir}"/model.*.pt
do
    m=$(basename "${x}")
    grep "$m" "$plot_file" || \
            (
                data=$(./scripts/evaluate.sh -g "${games}" "${x}")
                printf "%s\t%s\n" "$m" "${data}" | tee -a "${plot_file}"
            )
done
chmod a-w "${plot_file}"
