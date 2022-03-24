#!/bin/bash


# Parameter is a filename of a file containing the output of python3 print_elements_num.py -f /exports/eddie/scratch/s0093128/Data/*
file=${1}
# Optional second parameter is the amount of RAM to request for the job 32 is the default
mem=${2:-32}

# Required number of trials to be collected
#total_trials=40
total_trials=${3:-40}

file_id=${4:-eddie10}

type=($(cat "${file}" | grep -v "is not " | cut -f 2 -d ' ' | grep -oP '(NMDA|EC_LV_1)'))
coll_num=($(cat "${file}" | grep -v "is not " | cut -f 4 -d ' ' | grep -oE '([[:digit:]]+)'))
noise=($(cat "${file}" | grep -v "is not " | cut -f 2 -d ' ' | grep -oP '_noise.+Hz' | grep -oP '[0-9.]+'))
neurons_num=($(cat "${file}" | grep -v "is not " | cut -f 2 -d ' ' | grep -oP '_[0-9]+.npy' | grep -oP '[0-9]+'))

# Length of array
total=${#type[@]}

# Iterate over arrays by index
for (( i=0; i<=$(( $total -1 )); i++ ))
do
    model_type="${type[$i]}"
    if [ ${model_type} == "EC_LV_1" ]
    then
        model_type="full_1"
    fi

    echo "qsub ./run_trials_job_${mem}G.sh ${model_type} ${neurons_num[$i]} 300 $((${total_trials} - ${coll_num[$i]})) ${noise[$i]} ${file_id}"
    # Invoke the job to collect the missing trial data
    qsub ./run_trials_job_${mem}G.sh "${model_type}" "${neurons_num[$i]}" 300 "$((${total_trials} - ${coll_num[$i]}))" "${noise[$i]}" "${file_id}"
done

