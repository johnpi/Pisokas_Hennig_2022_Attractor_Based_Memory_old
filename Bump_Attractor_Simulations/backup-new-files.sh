#!/bin/bash

mkdir -p /exports/eddie/scratch/s0093128/Data/Backup2
sleep 20 # Wait 1s in case any files are being written to disk right now

# Copy files modified between backup-checkpoint and Backup2 to a temporal directory
#find /exports/eddie/scratch/s0093128/Data -maxdepth 1 -type f -newer /exports/eddie/scratch/s0093128/Data/Backup -exec cp '{}' /exports/eddie/scratch/s0093128/Data/Backup2/ \;
find /exports/eddie/scratch/s0093128/Data -maxdepth 1 -type f -newer /exports/eddie/scratch/s0093128/Data/backup-checkpoint -and ! -newer /exports/eddie/scratch/s0093128/Data/Backup2 -exec cp '{}' /exports/eddie/scratch/s0093128/Data/Backup2/ \;

# Update time stamp marker that we have copied everything up to this file modification time
touch /exports/eddie/scratch/s0093128/Data/backup-checkpoint

# Check if files are valid and move valid files to Backup
for f in /exports/eddie/scratch/s0093128/Data/Backup2/*; do python3 print_elements_num.py -f $f; done > output.txt 
for f in $(cat output.txt | grep -v 'is not access' | cut -f 2 -d ' '); do cp $f /exports/eddie/scratch/s0093128/Data/Backup/ ; done

# Remove temporal backup files
rm -rf /exports/eddie/scratch/s0093128/Data/Backup2/
