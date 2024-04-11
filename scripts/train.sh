set -e
set -x

while getopts ":g:m:o:" opt; do
    case $opt in
        g)
            GAMES="${OPTARG}"
            ;;
        m)
            MODEL="${OPTARG}"
            ;;
        o)
            OUT="${OPTARG}"
            ;;
        *)
            exit 1
    esac
done

shift "$((OPTIND-1))"

if [ -z "$GAMES" ]
then
    exit 1
fi
if [ -z "$MODEL" ]
then
    exit 1
fi
if [ -z "$OUT" ]
then
    exit 1
fi

if docker run --rm --gpus all hello-world >/dev/null 2>/dev/null; then
    GPU_ARGS="--gpus all";
else
    GPU_ARGS="";
fi

docker run --rm \
       $GPU_ARGS \
       --mount type=bind,src=$(pwd)/var/move_tensors,target=/var/move_tensors \
       --mount type=bind,src=$(pwd)/var/models,target=/var/models \
       td-gammon train \
       --move-tensors /var/move_tensors/current \
       --save-dir /var/models/${MODEL} \
       --out $OUT \
       --iterations $GAMES

