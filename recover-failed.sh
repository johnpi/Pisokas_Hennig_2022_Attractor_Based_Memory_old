#!/bin/bash

# Checks if any of the files in the provided file were invalid and attempts to recover them from backup. 

# the given list of files is created with something like this 
#  for f in /exports/eddie/scratch/s0093128/Data/Backup/collected_drift_trials_all_EC_LV_1_duration300s_noise*Hz_veddie10_256.npy; do python3 print_elements_num.py -f $f; done > output2.txt  &
# and when this script is invoked as 
#  recover-failed.sh output2.txt 
# it will double check the files in Data/ are invalid and will copy them over with the verisons in Backup/

# Note this script creates a temporal directory /exports/eddie/scratch/s0093128/Data/BackupTMP3
# and the file output-failed.txt


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

if [ -e /exports/eddie/scratch/s0093128/Data/BackupTMP3 ] 
then
    echo "Temporal directory /exports/eddie/scratch/s0093128/Data/BackupTMP2 already exists."
    echo "Aborting..."
    exit 1
fi


mkdir /exports/eddie/scratch/s0093128/Data/BackupTMP3/

# Make sure the input file does not contain a wildcard star which matches everything. It should contain only specific files
grep 'File /exports/eddie/scratch/s0093128/Data/Backup2/\* is not accessible' output.txt 2> /dev/null
if [ $? -ne 0 ]
then
    for f in $(cat "${1}" | grep 'is not' | cut -f 2 -d ' ')
    do
        cp $(echo $f | sed 's|Backup2/||') /exports/eddie/scratch/s0093128/Data/BackupTMP3/
    done
fi

echo "" > output-failed.txt
# Make sure the directory is not empty
ls /exports/eddie/scratch/s0093128/Data/BackupTMP3/*.npy 2> /dev/null
if [ $? -eq 0 ]
then
    for f in /exports/eddie/scratch/s0093128/Data/BackupTMP3/*.npy
    do
        python3 print_elements_num.py -f $f >> output-failed.txt
    done
fi

for f in $(cat output-failed.txt | grep 'is not' | cut -f 2 -d ' ')
do
    cp $(echo $f | sed 's|BackupTMP3/|Backup/|') /exports/eddie/scratch/s0093128/Data/
done

rm -rf /exports/eddie/scratch/s0093128/Data/BackupTMP3

