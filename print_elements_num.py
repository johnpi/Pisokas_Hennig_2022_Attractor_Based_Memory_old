# Prints the number of elements in the np.load file 

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

# Load libraries
import sys
import argparse # For command line argument parsing

import numpy as np


# New command line options set up
parser = argparse.ArgumentParser(description='Prints the number of elements in the numpy array stored in each of the provided files.')

# File to read
parser.add_argument('-f', '--file', type=str, nargs='+', dest='input_files', required=True,
                   help='Input filenames to read data from.')

# Parse the command line arguments
args = parser.parse_args()

input_files = args.input_files

for file in input_files:
    try:
        data = np.load(file, allow_pickle=True, encoding='bytes')
        print('File {} contains {} elements'.format(file, data.shape))
    except: 
        print('File %s is not accessible')
