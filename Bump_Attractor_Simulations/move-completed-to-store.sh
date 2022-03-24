#!/bin/bash

# Moves files that have 40 or 100 entries listed in the given file and moves
# them to the /exports/eddie/scratch/s0093128/Data/Completed/ directory. 
# Then removes the original copies from the /exports/eddie/scratch/s0093128/Data/ 
# and /exports/eddie/scratch/s0093128/Data/Backup directories and does 
# some crosschecking that they were copied correctly. 
#
# the list of files can be created with something like this 
#  for f in /exports/eddie/scratch/s0093128/Data/Backup/collected_drift_trials_all_EC_LV_1_duration300s_noise*Hz_veddie10_256.npy; do python3 print_elements_num.py -f $f; done > output2.txt  &
# and then invoke this script as 
#  move-completed-to-store.sh output2.txt 
#
# Note this script creates a temporal directory /exports/eddie/scratch/s0093128/Data/BackupTMP2


if [ "$#" == "0" ]
then
    echo "Expects a filename as argument. See comments in script for explanation."
    echo "Aborting..."
    exit 1
fi

if [ ! -e "$1" ]
then
    echo "Provided file does not exist."
    exit 1
fi

if [ -e /exports/eddie/scratch/s0093128/Data/BackupTMP2 ] 
then
    echo "Temporal directory /exports/eddie/scratch/s0093128/Data/BackupTMP2 already exists."
    echo "Aborting..."
    exit 1
fi

mkdir /exports/eddie/scratch/s0093128/Data/BackupTMP2/

for f in $(cat "$1" | grep -E '\((40|100),\)' | cut -f 2 -d ' ') 
do 
    echo "Processing $f ..."
    mv $(echo $f | sed 's|Backup2/|Backup/|') /exports/eddie/scratch/s0093128/Data/Completed/
    mv $(echo $f | sed 's|Backup/||') /exports/eddie/scratch/s0093128/Data/BackupTMP2/

    # Compares the copied file with the original to confirm it is correctly copied
    cmp $(echo $f | sed 's|Backup2/|BackupTMP2/|') $(echo $f | sed 's|Backup2/|Completed/|')
done

rm -rf /exports/eddie/scratch/s0093128/Data/BackupTMP2/


