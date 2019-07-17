#!/bin/bash

#
# INSTALLING BRIAN2 ON MACOSX
# ===========================

echo -n "Is pip3 installed? "
if python -m pip3 --version 2> /dev/null
then
  echo "YES"
else
  echo "NO"
  echo "Installing pip in the user account..."
  easy_install --user pip3
  python -m pip3 install --upgrade --user --no-binary :all: pip3
  echo 
fi

# Make sure we got g++ compiler installed
echo -n "Is a g++ compiler installed? "
if g++ --version 2> /dev/null
then
  echo "YES A g++ compiler is installed: OK"
else
  echo "NO"
  echo "Please, install a g++ compiler so that Brian2 runs faster."
  exit 1
fi

echo 
echo "Installing packages required by Brian2 and Brian2 itself..."
python -m pip3 install --upgrade --user --no-binary :all: Cython      && \
python -m pip3 install --upgrade --user --no-binary :all: nose        && \
python -m pip3 install --upgrade --user --no-binary :all: brian2      && \
python -m pip3 install --upgrade --user --no-binary :all: brian2tools && \
echo "Brian2 was installed successfully."

echo 
echo "Installing extras: Jupyter notebooks"
python -m pip3 install --upgrade --user --no-binary :all: jupyter

echo 
echo "To test the installation of Brian2 I run"
echo "\$ python3"
echo "and inside python invoke:"
echo ">>> import brian2"
echo ">>> brian2.test()"
echo "This will take some time and should print OK for all sets of tests."
echo

exit 0
