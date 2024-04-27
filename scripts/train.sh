set -e

function error_exit {
    echo "$@" >&2
    exit 1
}

while getopts ":g:h:o:e:a:l:c:d:f:" opt; do
    case $opt in
        f)
            fork="${OPTARG}"
            ;;
        c)
            continue="${OPTARG}"
            ;;
	d)
	    epc_db="${OPTARG}"
	    ;;
        g)
            games="${OPTARG}"
            ;;
        h)
            hidden="${OPTARG}"
            ;;
        o)
            if [ "${OPTARG}" == "4" ] || [ "${OPTARG}" == "6" ]
            then
                out="${OPTARG}"
            else
                error_exit "-o {out} must b 4 or 6" >&2
            fi
            ;;
        e)
            if [ "${OPTARG}" == "baine" ] || [ "${OPTARG}" == "baine_epc" ] || [ "${OPTARG}" == "tesauro" ]
            then
                encoding="${OPTARG}"
            else
                error_exit "-e {encoding} must be baine or tesauro" >&2
            fi
            ;;
        a)
            alpha="${OPTARG}"
            ;;
        l)
            lambda="${OPTARG}"
            ;;
        *)
            exit 1
    esac
done

shift "$((OPTIND-1))"

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

function train_fork {

    if [ -z "$games" ]
    then
        error_exit "pass -g {games}"
    fi
    if [ -z "$alpha" ]
    then
        error_exit "pass -a <alpha> variable"
    fi
    if [ -z "${fork}" ]
    then
        error_exit "pass -f <fork>"
    fi

    docker run --rm \
           "${gpu_args[@]}" \
           --mount "type=bind,src=$(pwd)/var/models,target=/var/models" \
	   "${epc_docker_args[@]}" \
           td-gammon train \
           ${force_cuda} \
           --fork "/${fork}" \
           --alpha "${alpha}" \
           --iterations "${games}" \
	   "${epc_db_args[@]}" \
           --save-dir "/${fork}-$(date +%s)"
}

function train_continue {
    if [ -z "${continue}" ]
    then
        error_exit "pass -c {directory}"
    fi
    docker run --rm \
           "${gpu_args[@]}" \
           --mount "type=bind,src=$(pwd)/var/models,target=/var/models" \
	   "${epc_docker_args[@]}" \
           td-gammon train \
           ${force_cuda} \
           --continue \
	   "${epc_db_args[@]}" \
           --save-dir "/${continue}"
}

function train_from_start {
    if [ -z "$games" ]
    then
        error_exit "set games variable"
    fi
    if [ -z "$alpha" ]
    then
        error_exit "pass -a <alpha> variable"
    fi
    if [ -z "$lambda" ]
    then
        error_exit "pass -l <lambda> variable"
    fi
    if [ -z "$hidden" ]
    then
        error_exit "pass -h <hidden> variable"
    fi
    if [ -z "$out" ]
    then
        error_exit "pass -o {4,6}"
    fi
    if [ -z "$encoding" ]
    then
        error_exit "set encoding variable"
    fi

    D=$(date +%s)
    model="$encoding-$hidden-$out-$games-$D"

    docker run --rm \
           "${gpu_args[@]}" \
           --mount "type=bind,src=$(pwd)/var/models,target=/var/models" \
	   "${epc_docker_args[@]}" \
           td-gammon train \
           ${force_cuda} \
           --alpha "${alpha}" \
           --lambda "${lambda}" \
           --save-dir "/var/models/${model}" \
           --out "${out}" \
           --encoding "${encoding}" \
           --hidden "${hidden}" \
	   "${epc_db_args[@]}" \
           --iterations "${games}"
}

if [ -n "${continue}" ]
then
    train_continue
elif [ -n "${fork}" ]
then
    train_fork
else
     train_from_start
fi
