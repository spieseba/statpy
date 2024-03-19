#!/bin/bash

python3 -m pip install --no-build-isolation --editable .
# requires python packages: meson-python meson ninja as build dependencies need to be available when importing lib