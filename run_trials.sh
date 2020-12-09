#!/bin/bash

#  run_trials.sh
#  
#
#  Created by John on 03/12/2020.
#
#  Runs the data collection on the local machine.
#

 vmem=16000000000 # In Bytes
 
# Initialise the environment modules
. /etc/profile.d/modules.sh

source ~/.bashrc

# Load Python
module load anaconda/5.3.1

conda activate Brian2

# $1 full or reduced network
# $2 the number of excitatory neurons
# $3 the duration of the simulation
# $4 the number of trials to run
# $5 the amount of neuronal noise
# Run the program
if [ "${1}" == "reduced" ]; then
   echo	"Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N $2 -t $4 -D $3 --neuronal_noise_Hz ${5:-2.3} -a ${vmem} -f Data/collected_drift_trials_all_EC_LV_reduced_2_duration$3s_noise${5:-2.3}Hz_veddie02_$2.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N "$2" -t "$4" -D "$3" --neuronal_noise_Hz "${5:-2.3}" -a "${vmem}" -f "Data/collected_drift_trials_all_EC_LV_reduced_2_duration$3s_noise${5:-2.3}Hz_veddie02_$2.npy"
fi

if [ "${1}" == "full" ]; then
   echo	"Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons.py -N $2 -t $4 -D $3 --neuronal_noise_Hz ${5:-2.3} -a ${vmem} -f Data/collected_drift_trials_all_EC_LV_duration$3s_noise${5:-2.3}Hz_veddie02_$2.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons.py -N "$2" -t "$4" -D "$3" --neuronal_noise_Hz "${5:-2.3}" -a "${vmem}" -f "Data/collected_drift_trials_all_EC_LV_duration$3s_noise${5:-2.3}Hz_veddie02_$2.npy"
fi

