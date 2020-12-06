#!/bin/bash

if [ "$#" -lt 1 ]; then
   echo "ERROR: Expects at least one argument."
   echo "Sends me an email if a file has not been modified for 300s or for the specified number of seconds."
   echo
   echo "USAGE:"
   echo "watch -n 60 $0 <FILE_TO_WATCH> <CHECK_EVERY_X_SEC>"
   exit 1
fi

file="${1}" && last_mod_date=$(stat --printf="%Y" "${file}") && now_date=$(date +%s) && diff=$((${now_date} - ${last_mod_date})) && [ ${diff} -ge ${2:-600} ] && echo "Complete file ${file} at $(date)" | mailx -s "Task complete: Eddie simulations finished" s0093128@ed.ac.uk && exit
