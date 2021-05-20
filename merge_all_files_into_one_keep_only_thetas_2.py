# -*- coding: utf-8 -*-

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

# Load libraries
import sys
import argparse # For command line argument parsing
import math
import numpy as np
import glob

from utility_functions import *
from Python_Libs.utility_functions import *


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
    
    for collected_data_file in collected_data_files:
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
            collected_trials_data = np.array([]) # Collected trials data records list

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
        plot_item = pick_time_series_list(collected_data_file, 
                                          stimulus_center_deg   = stimulus_center_deg,
                                          stimulus_width_deg    = None,
                                          sim_time_duration     = None,
                                          N_excitatory          = plot_key,
                                          synaptic_noise_amount = synaptic_noise_amount,
                                          unwrap_modulo_angles  = unwrap_modulo_angles
                                         )
        
        plot_items_dict[plot_key] = plot_item
    return plot_items_dict





def merge_file(output_file):
    """
        Reads and combines all the data records from multiple files 
        and stores them in a new file.
        output_file      : is the output filename.
        input_files_list : is a list of filenames to combine.
        
        
    """
    
    collected_trials_data = {} # Collected trials data records list

    
    collected_data_file_pattern = '/exports/eddie/scratch/s0093128/Data/Backup/collected_drift_trials_all_{:}_duration300s_noise{:}Hz_veddie*_{:}.npy'
    models_list           = ['NMDA', 'EC_LV_1']
    neurons_num_list      = [256, 512, 1024, 2048, 4096, 8192]
    plot_keys_list        = neurons_num_list
    poisson_firing_rate   = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
    stimulus_center_deg   = 180
    synaptic_noise_amount = 0
    
    # For each model type
    for model in models_list:
        collected_trials_data[model] = {}
        # For each neuronal noise level
        for poisson_neuron_noise in poisson_firing_rate:
            
            print('Get matching files: {:}'.format(collected_data_file_pattern.format(model, poisson_neuron_noise, '*')))
            collected_data_file_list = glob.glob(collected_data_file_pattern.format(model, poisson_neuron_noise, '*'))
            print('Found {:} files.'.format(collected_data_file_list))
            
            plot_items_dict = pick_net_size_data(collected_data_file_list, 
                                                 plot_keys_list, 
                                                 stimulus_center_deg = stimulus_center_deg, 
                                                 synaptic_noise_amount = synaptic_noise_amount,
                                                 unwrap_modulo_angles = True)
            
            # This is a dict with structure ['NMDA|EC_LV_1']['0.001']['1024'] = time_series_collection
            collected_trials_data[model][poisson_neuron_noise] = plot_items_dict
        
    # Save all data in the file
    np.save(output_file, collected_trials_data, allow_pickle=True)



# New command line options set up
parser = argparse.ArgumentParser(description='Merge the items of the ndarrays in the provided collect data files into one file.')

# File to write all data to
parser.add_argument('-f', '--file', type=str, dest='output_file', required=True,
                   help='Output filename to write all collected data to.')
#parser.add_argument('-i', '--input-files', type=str, nargs='+', dest='input_files', required=True,
#help='One or more filename of .npy files to read data from and combine them into one output file.')
#parser.add_argument('-m', '--max', type=int, dest='max_entities', required=False, default=None,
#help='Keep only the first [max] entries from each file and merge them to the output file.')

# Parse the command line arguments
args = parser.parse_args()

#input_files = args.input_files
output_file = args.output_file
#max_entities = args.max_entities

# Was
# merge_file(merged_data_filename, collected_data_files_list)
merge_file(output_file)