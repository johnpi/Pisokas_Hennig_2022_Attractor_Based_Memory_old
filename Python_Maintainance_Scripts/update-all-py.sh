#!/bin/bash

# Build and install from source. If it fails to build some package install it
# from binary by removing the --no-binary :all: switch

# This method stops on first error while installing packages
#python -m pip install --upgrade --user --no-binary :all: `python -m pip list --outdated | awk '!/Could not|ignored/ { print $1}' | grep -vE '^scipy|^matplotlib|^pyobjc-framework-Message|^pyOpenSSL|^Package|^----'`

# This method continues until the end even if some packages fail to update
PACKAGES_LIST=$(python -m pip list --outdated | awk '!/Could not|ignored/ { print $1}' | grep -vE '^Package|^----')
for item in ${PACKAGES_LIST[@]}
do
    python -m pip install --upgrade --user --no-binary :all: "${item}"
    # If not successful installing from source try binary
    if [ $? -ne 0 ]
    then
        python -m pip install --upgrade --user "${item}"
    fi
done