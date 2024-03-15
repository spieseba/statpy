#!/bin/bash

# update commit hash
new_commit_hash=$(git rev-parse HEAD)
sed -i "s/^COMMIT_HASH = .*/COMMIT_HASH = \"${new_commit_hash}\"/" ./src/statpy/database/core.py

# build wheel
python3 -m build

# find wheel file
wheel_file=$(ls dist/*.whl)

# install wheel
pip install --force-reinstall $wheel_file
