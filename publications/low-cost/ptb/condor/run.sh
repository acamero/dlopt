#!/bin/bash
PROGRAM="$1"
CONFIG="$2.json"
echo "./$PROGRAM --config $CONFIG --seed 0"
./$PROGRAM --config $CONFIG --seed 0


