#!/bin/bash

PIPE=pipe.txt
GNUBG_ERR=gnubg.err
GNUBG_OUT=gnubg.out
PY_ERR=py.err
PY_OUT=py.out
rm -f $PIPE $GNUBG_ERR $GNUBG_OUT $PY_ERR $PY_OUT && \
    touch $PIPE && \
    tail -f $PIPE \
        | gnubg -q -t 2>$GNUBG_ERR \
        | tee $GNUBG_OUT \
        | python -u src/player.py 2>$PY_ERR \
        | tee $PY_OUT \
                 >> $PIPE
