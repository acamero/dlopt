#!/bin/bash

SEED="$1"
PROBLEM="$2"
export LC_ALL=C
virtualenv --python=/usr/bin/python3.6 venv
. venv/bin/activate
easy_install dlopt-0.1-py3.6.egg
pip install -r requirements.txt
python random-search.py --seed=$SEED --verbose=1 --problem=$PROBLEM
deactivate
