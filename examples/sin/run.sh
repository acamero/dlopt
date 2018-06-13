#!/bin/bash

python ../../dlopt/tools/main.py --config mae-rand-samp-sin.json --verbose=1

python ../../dlopt/tools/main.py --config mae-optimization-sin.json --verbose=1
