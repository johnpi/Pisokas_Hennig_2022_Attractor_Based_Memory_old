# -*- coding: utf-8 -*-

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

# Load libraries
import sys
import argparse # For command line argument parsing

import numpy as np

from utility_functions import *

# List of input files to read data from
collected_data_files_list = [
#    'Data/collected_drift_trials.npy',
#    'Data/collected_drift_trials_v2_angles.npy',
#    'Data/collected_drift_trials_v2_noise.npy',
#    'Data/collected_drift_trials_v2_Ne512_noise.npy',
#    'Data/collected_drift_trials_v2_noise_Ne2048.npy',
#    'Data/collected_drift_trials_v2_Ne4096_noise.npy'
    'Data/collected_drift_trials_v3_all.npy',
    'Data/collected_drift_trials_v4_all.npy',    
]

# File to store all the data to
#merged_data_filename = 'Data/collected_drift_trials_v2_all.npy'
merged_data_filename = 'Data/collected_drift_trials_v3and4_all.npy'



def merge_file(output_file, input_files_list, max_entities=None):
    """
        Reads and combines all the data records from multiple files 
        and stores them in a new file.
        output_file      : is the output filename.
        input_files_list : is a list of filenames to combine.
        
        
    """
    
    collected_trials_data = np.array([]) # Collected trials data records list
    
    # For each input file
    for collected_data_file in input_files_list:
        # Try to load existing data if any otherwise create an empty collection
        try:
            collected_trials_data_input = np.load(collected_data_file, allow_pickle=True, encoding='bytes')
        except: 
            collected_trials_data_input = np.array([]) # Collected trials data records list
        
        if max_entities is not None and len(collected_trials_data_input) > max_entities:
            for i in range(max_entities):
                item = collected_trials_data_input[i]
                # Add new data record to the collected trials data
                collected_trials_data = np.append(collected_trials_data, item)
        else:
            for i,item in enumerate(collected_trials_data_input):
                # Add new data record to the collected trials data
                collected_trials_data = np.append(collected_trials_data, item)
    
    # Save all data in the file
    np.save(output_file, collected_trials_data, allow_pickle=True)



# Was
# merge_file(merged_data_filename, collected_data_files_list)


# New command line options set up
parser = argparse.ArgumentParser(description='Merge the items of the ndarrays in the provided collect data files into one file.')

# File to write all data to
parser.add_argument('-f', '--file', type=str, dest='output_file', required=True,
                   help='Output filename to write all collected data to.')
parser.add_argument('-i', '--input-files', type=str, nargs='+', dest='input_files', required=True,
help='One or more filename of .npy files to read data from and combine them into one output file.')
parser.add_argument('-m', '--max', type=int, dest='max_entities', required=False, default=None, 
help='Keep only the first [max] entries from each file and merge them to the output file.')

# Parse the command line arguments
args = parser.parse_args()

input_files = args.input_files
output_file = args.output_file
max_entities = args.max_entities

# Was
# merge_file(merged_data_filename, collected_data_files_list)
merge_file(output_file, input_files, max_entities)
