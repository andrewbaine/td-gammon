set -e
set -x

NOW=$(date -Iseconds)

N=$1

PORT=8901

PLAYER_1=andrewbaine
GNUBG=gnubg

LOG_FILE=gnubg.${NOW}.log

cat <<HERE | gnubg -q -t > ${LOG_FILE}
new session
set player andrewbaine external localhost:$PORT
set jacoby off
new session $N
export session text session.${NOW}.txt
HERE

p1=$(< ${LOG_FILE} grep "$PLAYER_1 wins a single game" ${LOG_FILE} | wc -l | perl -wpl -e 's|\s+||g')
p2=$(< ${LOG_FILE} grep "$PLAYER_1 wins a gammon"  ${LOG_FILE} | wc -l | perl -wpl -e 's|\s+||g')
p3=$(< ${LOG_FILE} grep "$PLAYER_1 wins a backgammon"  ${LOG_FILE} | wc -l | perl -wpl -e 's|\s+||g')
p4=$(< ${LOG_FILE} grep "$GNUBG wins a single game"  ${LOG_FILE} | wc -l | perl -wpl -e 's|\s+||g')
p5=$(< ${LOG_FILE} grep "$GNUBG wins a gammon"  ${LOG_FILE} | wc -l | perl -wpl -e 's|\s+||g')
p6=$(< ${LOG_FILE} grep "$GNUBG wins a backgammon"  ${LOG_FILE} | wc -l | perl -wpl -e 's|\s+||g')

echo "scale=4; (1 * $p1 + 2 * $p2 + 3 * $p3 - 1 * $p4 - 2 * $p5 - 3 * $p6) / $N" | bc
