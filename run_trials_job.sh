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

vmem=16000000000 # Available RAM in Bytes
 
# Defaults for optional parameters
VERSION="eddie02"
NOISE="2.3"

# Usage string
USAGE=$(cat <<- EOM
USAGE
    `basename $0` <MODEL> <NEURONS> <DURATION> <TRIALS> [NOISE] [VERSION] [TAU_M]

     MODEL      : The model to use: 'NMDA', 'EC_LV_1', 'SIMPLE', 'full', or 'reduced' network
     NEURONS    : The number of excitatory neurons
     DURATION   : The duration of the simulation in seconds
     TRIALS     : The number of trials to run
     NOISE      : The amount of neuronal noise (optional, default 2.3)
     VERSION    : The version code to use in the filenames (optional)
     TAU_M      : The neuronal membrane time constant in ms (optional, only used by the SIMPLE model)
EOM
)

# Check if the required arguments were given
if [ "$#" -lt "4" ]; then
    echo "ERROR"
    echo "  Expected at least 4 arguments."
    echo
    echo "${USAGE}"
    exit $E_BADARGS
fi

MODEL=${1}               # $1 : Model to use: full or reduced network
NEURONS=${2}             # $2 : the number of excitatory neurons
DURATION=${3}            # $3 : the duration of the simulation
TRIALS=${4}              # $4 : the number of trials to run
NOISE=${5:-${NOISE}}     # $5 : the amount of neuronal noise (optional, default 2.3)
VERSION=${6:-${VERSION}} # $6 : the version code to use in the filenames (optional)

# Initialise the environment modules
. /etc/profile.d/modules.sh

source ~/.bashrc

# Load Python
module load anaconda/5.3.1

conda activate Brian2

# Run the program
if [ "${MODEL}" == "reduced" ]; then
   echo	"Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_reduced_2_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_reduced_2_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "full" ]; then
   echo	"Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "full_1" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_1.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_1_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_1.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_1_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "NMDA" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons_NMDA_Neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_NMDA_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_NMDA_Neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_NMDA_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "SIMPLE" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} --tau_m ${TAU_M} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_SIMPLE_duration${DURATION}s_tau${TAU_M}ms_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" --tau_m "${TAU_M}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_SIMPLE_duration${DURATION}s_tau${TAU_M}ms_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi
