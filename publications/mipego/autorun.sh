#!/bin/bash

unzip bundle.zip
export LC_ALL=C
virtualenv --python=/usr/bin/python3.6 venv
. venv/bin/activate
easy_install bayesopt-0.1-py3.6.egg
easy_install dlopt-0.2-py3.6.egg
pip install -r requirements.txt

python rnn-arch-opt.py $@
deactivate
