#!/bin/sh
if [ $1 -eq 1 ]
then
    python3 a1.py $2 $3 $4
elif [ $1 -eq 2 ]
then
    python3 a2.py $2 $3 $4 $5
fi