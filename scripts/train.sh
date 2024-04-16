set -e
set -x

while getopts ":g:h:o:e:a:l:c:" opt; do
    case $opt in
        c)
            continue="${OPTARG}"
            ;;
        g)
            games="${OPTARG}"
            ;;
        h)
            hidden="${OPTARG}"
            ;;
        o)
            out="${OPTARG}"
            ;;
        e)
            encoding="${OPTARG}"
            ;;
        a)
            alpha_arg=" --alpha ${OPTARG} "
            ;;
        l)
            lambda_arg=" --lambda ${OPTARG} "
            ;;
        *)
            exit 1
    esac
done

shift "$((OPTIND-1))"

if docker run --rm --gpus all hello-world >/dev/null 2>/dev/null; then
    gpu_args="--gpus all";
else
    gpu_args="";
fi

function train_continue {
    docker run --rm \
           "${gpu_args}" \
           --mount "type=bind,src=$(pwd)/var/models,target=/var/models" \
           td-gammon train \
           --continue \
           --save-dir "/${continue}"
}

function train_from_start {
    if [ -z "$games" ]
    then
        echo "set games variable"
        exit 1
    fi
    if [ -z "$hidden" ]
    then
        echo "set hidden variable"
        exit 1
    fi
    if [ -z "$out" ]
    then
        echo "set out variable"
        exit 1
    fi
    if [ -z "$encoding" ]
    then
        echo "set encoding variable"
        exit 1
    fi
    
    D=$(date +%s)
    model="$encoding-$hidden-$out-$games-$D"

    docker run --rm \
           "${gpu_args}" \
           --mount "type=bind,src=$(pwd)/var/models,target=/var/models" \
           td-gammon train "${alpha_arg}" "${lambda_arg}" \
           --save-dir "/var/models/${model}" \
           --out "${out}" \
           --encoding "${encoding}" \
           --hidden "${hidden}" \
           --iterations "${games}"
}

if [ -n "${continue}" ]
then
    train_continue
else
    train_from_start
fi

