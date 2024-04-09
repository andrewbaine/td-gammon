#!/bin/bash


GNUBG_ERR=gnubg.err
GNUBG_OUT=gnubg.out
PY_ERR=py.err
PY_OUT=py.out

rm -f $GNUBG_ERR $GNUBG_OUT $PY_ERR $PY_OUT

touch $GNUBG_OUT $PY_OUT


python -u src/evaluate.py --load-model tmp/model.00001000.pt --games 100 \
       2>$PY_ERR \
       > $PY_OUT \
       < <(tail -f $GNUBG_OUT) \
    &


PYTHON_JOB="$!"

gnubg -q -t \
      2>$GNUBG_ERR \
      > $GNUBG_OUT \
      < <(tail -f $PY_OUT) \
     &

GNUBG_JOB="$!"

wait $PYTHON_JOB

A=$(grep "gnubg wins a backgammon" $GNUBG_OUT | wc -l)
B=$(grep "gnubg wins a gammon" $GNUBG_OUT | wc -l)
C=$(grep "gnubg wins a single game" $GNUBG_OUT | wc -l)

D=$(grep " wins a single game" $GNUBG_OUT | grep -v gnubg | wc -l)
E=$(grep " wins a gammon" $GNUBG_OUT | grep -v gnubg | wc -l)
F=$(grep " wins a backgammon" $GNUBG_OUT | grep -v gnubg | wc -l)

CALCULATION="(-3 * $A - 2 * $B - $C + $D + 2 * $E + 3 * $F) / ($A + $B + $C + $D + $E + $F)"
PPG=$(echo "scale=2; $CALCULATION" | bc)

echo "$PPG $A $B $C $D $E $F"
