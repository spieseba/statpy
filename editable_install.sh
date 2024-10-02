#!/bin/bash

# requires python packages: meson-python meson ninja as build dependencies need to be available when importing lib
python3 -m pip install meson-python meson ninja
python3 -m pip install --no-build-isolation --editable .
