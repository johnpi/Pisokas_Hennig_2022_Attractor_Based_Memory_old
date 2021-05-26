#!/bin/bash

#cat output.txt | grep 'is not' >> output-backup-is-not.txt
#mv output-backup-is-not.txt output-backup-is-not-old.txt
#cat output.txt | grep 'is not' >> output-backup-is-not.txt

# For every file in the temporal Backup2/ that was invalid check if the copy in Backup/ is valid
for f in $(cat output.txt | grep 'is not' | cut -f 2 -d ' '); do python3 print_elements_num.py -f $(echo $f | sed 's|Backup2/|Backup/|'); done > output-2.txt

# For every file in temporal Backup2/ that was invalid check if the copy in Data/ is value
mkdir -p /exports/eddie/scratch/s0093128/Data/Backup3/
for f in $(cat output.txt | grep 'is not' | cut -f 2 -d ' '); do cp $(echo $f | sed 's|Backup2/||') /exports/eddie/scratch/s0093128/Data/Backup3/ ; python3 print_elements_num.py -f $(echo $f | sed 's|Backup2/|Backup3/|'); done > output-1.txt


# IFS= (or IFS='') prevents leading/trailing whitespace from being trimmed.
# -r prevents backslash escapes from being interpreted.
while IFS= read -r line; do
    echo "Text read from file: $line"
done < my_filename.txt


