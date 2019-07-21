# -*- coding: utf-8 -*-

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

# Load libraries
import sys
import numpy as np

from utility_functions import *

# List of input files to read data from
collected_data_files_list = [
    'Data/collected_drift_trials.npy',
    'Data/collected_drift_trials_v2_angles.npy',
    'Data/collected_drift_trials_v2_noise.npy',
    'Data/collected_drift_trials_v2_Ne512_noise.npy',
    'Data/collected_drift_trials_v2_noise_Ne2048.npy',
    'Data/collected_drift_trials_v2_Ne4096_noise.npy'
]

# File to store all the data to
merged_data_filename = 'Data/collected_drift_trials_v2_all.npy'


def merge_file(collected_data_files_list, merged_data_filename):
    """
        Reads and combines all the data records from multiple files 
        and stores them in a new file. 
    """
    
    collected_trials_data = np.array([]) # Collected trials data records list
    
    # For each input file
    for collected_data_file in collected_data_files_list:
        # Try to load existing data if any otherwise create an empty collection
        try:
            collected_trials_data_input = np.load(collected_data_file, allow_pickle=True, encoding='bytes')
        except: 
            collected_trials_data_input = np.array([]) # Collected trials data records list

        for i,item in enumerate(collected_trials_data_input):
            # Add new data record to the collected trials data
            collected_trials_data = np.append(collected_trials_data, item)
    
    # Save all data in the file
    np.save(merged_data_filename, collected_trials_data, allow_pickle=True)



merge_file(collected_data_files_list, merged_data_filename)
