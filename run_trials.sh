#!/bin/bash

#  run_trials.sh
#  
#
#  Created by John on 03/12/2020.
#
#  Runs the data collection on the local machine.
#

vmem=16000000000 # Available RAM in Bytes
 
# Defaults for optional parameters
VERSION="eddie02"
NOISE="2.3"

# Usage string
USAGE=$(cat <<- EOM
USAGE
    `basename $0` <MODEL> <NEURONS> <DURATION> <TRIALS> [NOISE] [VERSION]

     MODEL      : The model to use: 'full', 'reduced' or 'NMDA' network
     NEURONS    : The number of excitatory neurons
     DURATION   : The duration of the simulation in seconds
     TRIALS     : The number of trials to run
     NOISE      : The amount of neuronal noise (optional, default 2.3)
     VERSION    : The version code to use in the filenames (optional)
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

# Run the program
if [ "${MODEL}" == "reduced" ]; then
   echo	"Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f Data/collected_drift_trials_all_EC_LV_reduced_2_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "Data/collected_drift_trials_all_EC_LV_reduced_2_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "full" ]; then
   echo	"Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f Data/collected_drift_trials_all_EC_LV_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "Data/collected_drift_trials_all_EC_LV_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "full_1" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_1.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f Data/collected_drift_trials_all_EC_LV_1_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_1.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "Data/collected_drift_trials_all_EC_LV_1_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "NMDA" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons_NMDA_Neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f Data/collected_drift_trials_all_NMDA_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_NMDA_Neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "Data/collected_drift_trials_all_NMDA_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi
