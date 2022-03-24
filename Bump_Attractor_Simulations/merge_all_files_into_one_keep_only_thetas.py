# -*- coding: utf-8 -*-

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

# Load libraries
import sys
import os
import argparse # For command line argument parsing
import math
import numpy as np
import glob

from utility_functions import *
from Python_Libs.utility_functions import *

DEBUG = False

def dispertion_of_absolute_deviation(expected_value, time_series_list):
    """
        Returns a measure of dispertion of the absolute deviation (MAD) of time series produced 
        by multiple trials at each point in time. That is a measure of 
        dispersion over time. 
        $MAD = \frac{1}{N} \sum_{i=0}^{N} \abs{x_{i} - mean(X)}$ 
        MAD is a better measure than standard deviation (SD). 
        
        expected_value   : The expected value for MAD normally is the mean of 
                           the sample. That is the mean value of all time series 
                           at that point in time. However you can provide 
                           any expected value here.
        time_series_list : A list or np.array containing np.arrays of several 
                           time series. All time series must have the same 
                           number of items. 
        Returns          : A time series with elements the dispertion of the absolute 
                           deviations at each point in time. It has the same number 
                           of items as the contained time series. 
    """
    
    # The absolute value of the differences of the time series samples from the expected value
    
    abs_diff_list = []
    # Get the absolute deviation of each item of the series from the expected value
    for ts in time_series_list:
        abs_diff = np.abs(np.ones(len(ts)) * expected_value - ts)
        abs_diff_list.append(abs_diff)
    
    # Get the mean absolute deviation across all time series at each point in time
    abs_diff_mean = np.std(abs_diff_list, axis=0)
    
    return abs_diff_mean



def pick_time_series_list(collected_data_file, 
                          stimulus_center_deg   = None,
                          stimulus_width_deg    = None,
                          sim_time_duration     = None,
                          N_excitatory          = None,
                          synaptic_noise_amount = None,
                          tau_excit             = None, 
                          tau_inhib             = None, 
                          unwrap_modulo_angles  = False
                          ):
    # Example how to create an array of timestamps spaced by snapshot_interval in the interval of interest.
    t_window_width      = 200*ms # was 100*ms
    snapshot_interval   = 100*ms

    theta_ts_list = [] # To store the time series

    time_series_collection = dict()
    
    if not isinstance(collected_data_file, list):
        collected_data_files = [collected_data_file]
    else:
        collected_data_files = collected_data_file
    
    if not DEBUG: print('      ', end='')
    for collected_data_file in collected_data_files:
        if DEBUG: print('      Processing file:', collected_data_file)
        print('.', end='')
        sys.stdout.flush()
        # Try to load existing data if any otherwise create an empty collection
        try:
            #collected_trials_data = np.load(collected_data_file, allow_pickle=True, encoding='bytes')
            # Load only records with specific values
            collected_trials_data = pick_data_samples(collected_data_file, 
                                    stimulus_center_deg   = stimulus_center_deg,
                                    stimulus_width_deg    = stimulus_width_deg,
                                    sim_time_duration     = sim_time_duration,
                                    N_excitatory          = N_excitatory,
                                    synaptic_noise_amount = synaptic_noise_amount, 
                                    tau_excit             = tau_excit, 
                                    tau_inhib             = tau_inhib, 
                                    operator              = 'and'
                                 )
        except: 
            print('      Exception while running pick_data_samples()')
            collected_trials_data = np.array([]) # Collected trials data records list
        
        if DEBUG: print('        Got len(collected_trials_data)', len(collected_trials_data))
        
        # We use enumerate to add a count to the iterated items of the iterator
        for i, item in enumerate(collected_trials_data):
            # Simulation set up info
            stimulus_center_deg              = item['stimulus_center_deg']
            stimulus_width_deg               = item['stimulus_width_deg']
            t_stimulus_start                 = item['t_stimulus_start']
            t_stimulus_duration              = item['t_stimulus_duration']
            sim_time_duration                = item['sim_time_duration']
            N_excitatory                     = item['N_excitatory']
            N_inhibitory                     = item['N_inhibitory']
            weight_scaling_factor            = item['weight_scaling_factor']

            # Data
            spike_monitor_excit_spike_trains = item['spike_monitor_excit']
            idx_monitored_neurons_excit      = item['idx_monitored_neurons_excit']
            spike_monitor_inhib_spike_trains = item['spike_monitor_inhib']
            idx_monitored_neurons_inhib      = item['idx_monitored_neurons_inhib']
            t_window_width                   = item['t_window_width']
            snapshot_interval                = item['snapshot_interval']
            theta_ts                         = item['theta_ts']
            t_snapshots                      = item['t_snapshots']

            # The newest version of get_theta_time_series_vec_add(...) returns a tuple (r_time_series, theta_time_series) 
            # Earlier versions were returning only theta_time_series
            if isinstance(theta_ts, tuple):
                theta_ts = theta_ts[1]

            #t_snapshots = range(
            #    int(math.floor((t_stimulus_start+t_stimulus_duration)/ms)),  # lower bound
            #    int(math.floor((sim_time_duration-t_window_width/2)/ms)),  # Subtract half window. Avoids an out-of-bound error later.
            #    int(round(snapshot_interval/ms))  # spacing between time stamps
            #    )*ms

            #theta_ts = get_theta_time_series_vec_add(spike_monitor_excit_spike_trains, idx_monitored_neurons_excit, N_excitatory, t_snapshots, t_window_width)

            # Do not limit angles to 0-360 degrees. Convert to increasing or decreasing non wrapping numbers.
            if unwrap_modulo_angles:
                theta_ts = unwrap_modulo_time_series(theta_ts, modulo = 360)

            # Store the time series
            theta_ts_list.append(theta_ts)
    
    if len(theta_ts_list) > 0:
        # Calculate the Mean Absolute Deviations (MAD)
        # TEMP theta_ts_abs_diff_mean = mean_absolute_deviation(expected_value=stimulus_center_deg, time_series_list=theta_ts_list)
        theta_ts_abs_diff_mean = median_absolute_deviation(expected_value=stimulus_center_deg, time_series_list=theta_ts_list)
        theta_ts_abs_diff_std = dispertion_of_absolute_deviation(expected_value=stimulus_center_deg, time_series_list=theta_ts_list)

        time_series_collection['theta_ts_list']          = theta_ts_list
        time_series_collection['theta_ts_abs_diff_mean'] = theta_ts_abs_diff_mean
        time_series_collection['theta_ts_abs_diff_std']  = theta_ts_abs_diff_std
        time_series_collection['t_snapshots']            = t_snapshots
        time_series_collection['stimulus_center_deg']    = stimulus_center_deg
        time_series_collection['N_excitatory']           = N_excitatory
        time_series_collection['idx_monitored_neurons_excit'] = idx_monitored_neurons_excit

        return time_series_collection
    else:
        return None


def pick_net_size_data(collected_data_file, N_excitatory_list, stimulus_center_deg = 180, synaptic_noise_amount = 0, unwrap_modulo_angles = False):
    # Plot collected time series for different stimulus angle locations
    plot_keys_list = N_excitatory_list
    plot_items_dict = dict()
    num_of_plot_keys = len(plot_keys_list)

    for i,plot_key in enumerate(plot_keys_list):
        print('    Look for N_excitatory =', plot_key)
        plot_item = pick_time_series_list(collected_data_file, 
                                          stimulus_center_deg   = stimulus_center_deg,
                                          stimulus_width_deg    = None,
                                          sim_time_duration     = None,
                                          N_excitatory          = plot_key,
                                          synaptic_noise_amount = synaptic_noise_amount,
                                          unwrap_modulo_angles  = unwrap_modulo_angles
                                         )
        if plot_item is not None:
            print('    Got len(plot_item[theta_ts_list])', len(plot_item['theta_ts_list']))
        else: 
            print('    plot_item is empty')
        plot_items_dict[plot_key] = plot_item
    return plot_items_dict





def merge_file(output_file, input_directory, unwrap_angles, filename_template_n):
    """
        Reads and combines all the data records from multiple files 
        and stores them in a new file.
        output_file      : is the output filename.
        input_files_list : is a list of filenames to combine.
        
        
    """
    
    collected_trials_data = {} # Collected trials data records list

    #path = './Data/'
    #path = '/Volumes/WD Elements 25A3 Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/Data/Completed/'
    #path = '/exports/eddie/scratch/s0093128/Data/Backup/'
    path = input_directory
    
    # Now selected from the command line
    if filename_template_n == 1: # Process files containing ['NMDA', 'EC_LV_1'] simulations
        filename_template = 'collected_drift_trials_all_{:}_duration300s_noise{:}Hz_veddie*_{:}.npy'
    elif filename_template_n == 2: # Process files containing ['SIMPLE'] simulations
        filename_template = 'collected_drift_trials_all_{:}_duration300s_tau{:}_noise{:}Hz_veddie*_{:}.npy'
    elif filename_template_n == 3: # Process files containing ['NMDA-SHIFT'] simulations
        filename_template = 'collected_drift_trials_all_{:}*_duration300s_noise{:}Hz_veddie*_{:}.npy'
        #filename_template = 'collected_drift_trials_all_{:}*_duration60s_noise{:}Hz_veddie*_{:}.npy'
    else:
        print('ERROR: Not acceptable value given filename_template=', filename_template)
        exit(1)
    
    collected_data_file_pattern = os.path.join(path, filename_template)
    
    # Default values
    neurons_num_list          = [128, 256, 512, 1024, 2048, 4096, 8192]
    poisson_firing_rate       = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
    models_list               = ['NMDA', 'EC_LV_1']
    neuron_time_constants     = ['complex']
    # Then modify if filename_template parameter is given
    if filename_template_n   == 1: # Process files containing ['NMDA', 'EC_LV_1'] simulations
        models_list           = ['NMDA', 'EC_LV_1']
        neuron_time_constants = ['complex']
    elif filename_template_n == 2: # Process files containing ['SIMPLE'] simulations
        neurons_num_list      = [256, 512]
        poisson_firing_rate   = [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.010, 0.100]
        poisson_firing_rate   = [1.0]
        poisson_firing_rate   = [1.4]
        poisson_firing_rate   = [1.4, 0.15, 0.3, 0.05]
        models_list           = ['SIMPLE']
        models_list           = ['SIMPLE-TAU2']
        models_list           = ['NMDA-TAU']
        neuron_time_constants = ['10ms', '100ms', '1000ms', '10000ms', '100000ms']
        neuron_time_constants = ['0.5ms', '1ms', '5ms', '10ms', '20ms', '30ms', '40ms', '50ms', '60ms', '70ms', '80ms', '90ms', '100ms']
    elif filename_template_n == 3: # Process files containing ['NMDA-SHIFT'] simulations (NMDA with bump shifting)
        models_list           = ['NMDA-SHIFT']
        models_list           = ['NMDA-SHIFT-0.001', 'NMDA-SHIFT-0.0005', 'NMDA-SHIFT-0.0001', 'NMDA-SHIFT-0.00001']
        neuron_time_constants = ['complex']
    else:
        pass
    
    stimulus_center_deg   = 180
    synaptic_noise_amount = 0
    
    # For each model type
    for model in models_list:
        print(model)
        collected_trials_data[model] = {}
        # For each neuronal noise level
        for poisson_neuron_noise in poisson_firing_rate:
            print('  poisson_neuron_noise', poisson_neuron_noise)
            collected_trials_data[model][poisson_neuron_noise] = {}
            for neuron_time_constant in neuron_time_constants:
                if filename_template_n == 2: # Process files containing ['SIMPLE'] simulations
                    print('  Get matching files: {:}'.format(collected_data_file_pattern.format(model, neuron_time_constant, poisson_neuron_noise, '*')))
                    collected_data_file_list = glob.glob(collected_data_file_pattern.format(model, neuron_time_constant, poisson_neuron_noise, '*'))
                else: # Process files containing ['NMDA', 'EC_LV_1', 'NMDA-SHIFT'] simulations
                    print('  Get matching files: {:}'.format(collected_data_file_pattern.format(model, poisson_neuron_noise, '*')))
                    collected_data_file_list = glob.glob(collected_data_file_pattern.format(model, poisson_neuron_noise, '*'))
                if DEBUG: print('  Found {:} files.'.format(collected_data_file_list))
                print('  Found {:} files.'.format(len(collected_data_file_list)))
            
                if DEBUG: print('  Call pick_net_size_data()')
                plot_items_dict = pick_net_size_data(collected_data_file_list, 
                                                     neurons_num_list, 
                                                     stimulus_center_deg = stimulus_center_deg, 
                                                     synaptic_noise_amount = synaptic_noise_amount,
                                                     unwrap_modulo_angles = unwrap_angles)
            
                # This is a dict with structure ['NMDA|EC_LV_1'][0.001]['complex|10ms']['1024'] = time_series_collection
                collected_trials_data[model][poisson_neuron_noise][neuron_time_constant] = plot_items_dict
        
    # Save all data in the file
    np.save(output_file, collected_trials_data, allow_pickle=True)



# New command line options set up
parser = argparse.ArgumentParser(description='Merge the items of the ndarrays in the provided collect data files into one file.')

# File to write all data to
parser.add_argument('-o', '--output-file', type=str, dest='output_file', required=True,
                   help='Output filename to write all combined data to.')
parser.add_argument('-i', '--input-directory', type=str, dest='input_directory', required=True,
help='One directory path to .npy files to read data from and combine them into the output file.')
parser.add_argument('-u', '--unwrap-angles', action='store_true', dest='unwrap_angles', required=False, default=False,
help='Switch: If provided heading angles are unwrapped using modulo 360 so after 360 is 361 and so on. Default is to not unwrap heading values so after 360 is 0.')
parser.add_argument('-t', '--filename-template', type=int, dest='filename_template', required=True, choices=[1, 2, 3], 
help='Specifies the filename template to use:\n 1 for "collected_drift_trials_all_{:}_duration300s_noise{:}Hz_veddie*_{:}.npy"\n 2 for "collected_drift_trials_all_{:}_duration300s_tau{:}_noise{:}Hz_veddie*_{:}.npy"\n 3 for "collected_drift_trials_all_{:}_duration300s_noise{:}Hz_veddie*_{:}.npy" with model NMDA-SHIFT')
#parser.add_argument('-i', '--input-files', type=str, nargs='+', dest='input_files', required=True,
#help='One or more filename of .npy files to read data from and combine them into one output file.')
#parser.add_argument('-m', '--max', type=int, dest='max_entities', required=False, default=None,
#help='Keep only the first [max] entries from each file and merge them to the output file.')

# Parse the command line arguments
args = parser.parse_args()

#input_files = args.input_files
input_directory = args.input_directory
output_file = args.output_file
#max_entities = args.max_entities
unwrap_angles = args.unwrap_angles
filename_template = args.filename_template

# Was
# merge_file(merged_data_filename, collected_data_files_list)
merge_file(output_file, input_directory, unwrap_angles, filename_template)
