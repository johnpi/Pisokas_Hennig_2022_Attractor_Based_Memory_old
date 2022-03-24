#!/bin/bash

for i in $(qstat | sed -E 's/[[:space:]]+/ /' | cut -f 2 -d ' '); do qstat -j $i | grep job_args ; done | sort

