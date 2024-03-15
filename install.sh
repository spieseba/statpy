#!/bin/bash

# get git hash
commit_hash=$(git rev-parse HEAD)

# build wheel
python3 -m build

# find wheel file
wheel_file=$(ls dist/*.whl)

# install wheel
pip install --force-reinstall $wheel_file
