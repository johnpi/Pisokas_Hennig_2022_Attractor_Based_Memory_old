#!/bin/bash
file="${1}" && last_mod_date=$(stat --printf="%Y" "${file}") && now_date=$(date +%s) && diff=$((${now_date} - ${last_mod_date})) && [ ${diff} -ge 600 ] && echo "Complete file ${file} at $(date)" | mailx -s "Task complete: Eddie simulations finished" s0093128@ed.ac.uk && exit
