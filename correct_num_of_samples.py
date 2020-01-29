# -*- coding: utf-8 -*-

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

# Load libraries
import sys
import numpy as np
from brian2 import *
from neurodynex.working_memory_network import wm_model
from neurodynex.working_memory_network import wm_model_modified

from utility_functions import *

def correct_file(collected_data_file,
                 t_window_width      = 200*ms,
                 snapshot_interval   = 100*ms
                ):
    
    collected_trials_data = np.array([]) # Collected trials data records list
    
    # Try to load existing data if any otherwise create an empty collection
    try:
        collected_trials_data_input = np.load(collected_data_file, allow_pickle=True, encoding='bytes')
    except: 
        collected_trials_data_input = np.array([]) # Collected trials data records list
    
    for i,item in enumerate(collected_trials_data_input):

        print('    Processing record: ', i)
        
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
        
        if 't_window_width' in item:
            t_window_width = item['t_window_width']
        
        if 'snapshot_interval' in item:
            snapshot_interval = item['snapshot_interval']
        
        if 'synaptic_noise_amount' not in item:
            item['synaptic_noise_amount'] = 0.0
        
        if 'theta_ts' in item:
            theta_ts                     = item['theta_ts']
        else:
            theta_ts = np.array([])

        if 't_snapshots' in item:
            t_snapshots                  = item['t_snapshots']
        else:
            t_snapshots = range(
                int(math.floor((t_stimulus_start+t_stimulus_duration)/ms)),  # lower bound
                int(math.floor((sim_time_duration-t_window_width/2)/ms)),  # Subtract half window. Avoids an out-of-bound error later.
                int(round(snapshot_interval/ms))  # spacing between time stamps
                )*ms
        
        print('Checking validity ', t_snapshots.shape[0], ' != ', len(theta_ts))
        if  (isinstance(theta_ts, list)  and t_snapshots.shape[0] != len(theta_ts)) or 
            (isinstance(theta_ts, tuple) and t_snapshots.shape[0] != len(theta_ts[1])):
            theta_ts = get_theta_time_series_vec_add(spike_monitor_excit_spike_trains, 
                                                     idx_monitored_neurons_excit, 
                                                     len(idx_monitored_neurons_excit), # Instead of N_excitatory because
                                                                                       # wm_model_modified.simulate_wm returns 
                                                                                       # in idx_monitored_neurons_excit at most 
                                                                                       # 1024 neurons 
                                                     t_snapshots, 
                                                     t_window_width)

        item['t_window_width']    = t_window_width
        item['snapshot_interval'] = snapshot_interval
        item['theta_ts']          = theta_ts
        item['t_snapshots']       = t_snapshots


        # Add new data record to the collected trials data
        collected_trials_data = np.append(collected_trials_data, item)

    # Save all data in the file
    np.save(collected_data_file, collected_trials_data, allow_pickle=True)




if len(sys.argv) > 1:
    collected_data_file = str(sys.argv[1])
    print('Processing file: ', collected_data_file)
    correct_file(collected_data_file)
else:
    print('Error: Please specify a valid .npy file name to process.')

