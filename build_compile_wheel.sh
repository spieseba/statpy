#!/bin/bash
rm -rf ./dist

# build wheel
python3 -m build

# find wheel file
wheel_file=$(ls dist/*.whl)

# install wheel
pip install --force-reinstall $wheel_file
