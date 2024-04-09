#!/bin/bash


GNUBG_ERR=gnubg.err
GNUBG_OUT=gnubg.out
PY_ERR=py.err
PY_OUT=py.out

rm -f $GNUBG_ERR $GNUBG_OUT $PY_ERR $PY_OUT

touch $GNUBG_OUT $PY_OUT


python -u src/player.py \
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

wait $GNUBG_JOB $PYTHON_JOB

grep wins $GNUBG_OUT
