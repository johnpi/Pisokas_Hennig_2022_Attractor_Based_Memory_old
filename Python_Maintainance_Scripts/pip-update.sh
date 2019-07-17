#!/bin/bash

python -m pip install --upgrade pip

# If you want to do it only for the current user account
# python -m pip install --user --upgrade pip
python -m pip install --upgrade --user --no-binary :all: pip