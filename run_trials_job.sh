#!/bin/sh

#  run_trials_job.sh
#  
#
#  Created by John on 03/12/2020.
#  
# Grid Engine options (lines prefixed with #$)
#$ -N run_trials_job
#$ -cwd
#$ -l h_vmem=32G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
## $ -l h_rt=96:00:00
#  runtime limit of 96 hours: -l h_rt
#  memory limit of 128 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda/5.3.1

conda activate Brian2

# $1 the number of excitatory neurons
# $2 the duration of the simulation
# $3 the number of trials to run
# Run the program
python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N "$1" -t "$3" -D "$2" -f "Data/collected_drift_trials_all_EC_LV_reduced_2_duration300s_veddie01_$1.npy"
