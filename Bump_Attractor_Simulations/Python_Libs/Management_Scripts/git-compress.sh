#!/bin/bash


#
# Reduce the size of the .git directory.
# Run this in the root git directory.
#

git reflog expire --all --expire=now
git gc --prune=now --aggressive

