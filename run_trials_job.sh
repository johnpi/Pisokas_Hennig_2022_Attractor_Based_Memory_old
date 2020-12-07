#!/bin/bash

#  run_trials_job.sh
#  
#
#  Created by John on 03/12/2020.
#  
# Grid Engine options (lines prefixed with #$)
#$ -N run_trials_job
#$ -cwd
#$ -l h_vmem=24G
## $ -l h_rt=48:00:00
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
## $ -l h_rt=72:00:00
#  runtime limit of 72 hours: -l h_rt
#  memory limit of 128 Gbyte: -l h_vmem

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
   echo	"Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N $2 -t $4 -D $3 --neuronal_noise_Hz ${5:-2.3} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_reduced_2_duration$3s_noise${5:-2.3}Hz_veddie02_$2.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N "$2" -t "$4" -D "$3" --neuronal_noise_Hz "${5:-2.3}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_reduced_2_duration$3s_noise${5:-2.3}Hz_veddie02_$2.npy"
fi

if [ "${1}" == "full" ]; then
   echo	"Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons.py -N $2 -t $4 -D $3 --neuronal_noise_Hz ${5:-2.3} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_duration$3s_noise${5:-2.3}Hz_veddie02_$2.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons.py -N "$2" -t "$4" -D "$3" --neuronal_noise_Hz "${5:-2.3}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_duration$3s_noise${5:-2.3}Hz_veddie02_$2.npy"
fi

