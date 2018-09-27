#!/bin/bash

# create virtual environment
virtualenv --python=python3 devenv/
source devenv/bin/activate
# install dependencies
pip install -r ../../../requirements.txt
cd ../../../
# install dlopt package
python setup.py sdist
package=$(ls dist/*.gz | tail -n 1)
pip install $package
cd -

# build binaries
pyinstaller main.spec
./dist/main --config config/test-dist.json
