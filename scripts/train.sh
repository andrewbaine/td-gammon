set -e
set -x

while getopts ":g:h:o:e:" opt; do
    case $opt in
        g)
            GAMES="${OPTARG}"
            ;;
        h)
            HIDDEN="${OPTARG}"
            ;;
        o)
            OUT="${OPTARG}"
            ;;
        e)
            ENCODING="${OPTARG}"
            ;;
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
if [ -z "$HIDDEN" ]
then
    echo "set HIDDEN variable"
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

D=$(date +%s)
MODEL="$ENCODING-$HIDDEN-$OUT-$GAMES-$D"

docker run --rm \
       $GPU_ARGS \
       --mount type=bind,src=$(pwd)/var/move_tensors,target=/var/move_tensors \
       --mount type=bind,src=$(pwd)/var/models,target=/var/models \
       td-gammon train \
       --move-tensors /var/move_tensors/current \
       --save-dir /var/models/${MODEL} \
       --out $OUT \
       --encoding $ENCODING \
       --hidden $HIDDEN \
       --iterations $GAMES

