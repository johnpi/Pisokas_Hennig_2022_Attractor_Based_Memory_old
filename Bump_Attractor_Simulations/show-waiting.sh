#!/bin/bash

for i in $(qstat | grep ' qw ' | cut -f 3 -d ' '); do qstat -j $i; done | grep submit_cmd


