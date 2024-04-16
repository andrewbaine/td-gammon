set -e
set -x

alpha="0.05"
lambda="1.0"

while getopts ":g:h:o:e:a:l:c:f:" opt; do
    case $opt in
        f)
            fork="${OPTARG}"
            ;;
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
            alpha="${OPTARG} "
            ;;
        l)
            lambda="${OPTARG} "
            ;;
        *)
            exit 1
    esac
done

shift "$((OPTIND-1))"

declare -a gpu_args
if docker run --rm --gpus all hello-world >/dev/null 2>/dev/null; then
    read -r -a gpu_args < <(echo "--gpus all")
fi

function train_fork {

    if [ -z "$games" ]
    then
        echo "set games variable"
        exit 1
    fi

    docker run --rm \
           "${gpu_args[@]}" \
           --mount "type=bind,src=$(pwd)/var/models,target=/var/models" \
           td-gammon train \
           --fork "/${fork}" \
           --alpha "${alpha}" \
           --lambda "${lambda}" \
           --iterations "${games}" \
           --save-dir "/${fork}-$(date +%s)"
}

function train_continue {
    docker run --rm \
           "${gpu_args[@]}" \
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
           "${gpu_args[@]}" \
           --mount "type=bind,src=$(pwd)/var/models,target=/var/models" \
           td-gammon train \
           --alpha "${alpha}" \
           --lambda "${lambda}" \
           --save-dir "/var/models/${model}" \
           --out "${out}" \
           --encoding "${encoding}" \
           --hidden "${hidden}" \
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
