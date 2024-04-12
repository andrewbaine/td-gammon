set -e
set -x

while getopts ":g:m:o:e:" opt; do
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
        e)
            ENCODING="${OPTARG}"
        *)
            exit 1
    esac
done

shift "$((OPTIND-1))"

if [ -z "$GAMES" ]
then
    echo "set GAMES variable"
    exit 1
fi
if [ -z "$MODEL" ]
then
    echo "set MODEL variable"
    exit 1
fi
if [ -z "$OUT" ]
then
    echo "set OUT variable"
    exit 1
fi
if [ -z "$ENCODING" ]
then
    echo "set ENCODING variable"
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
       --encoding $ENCODING \
       --iterations $GAMES

