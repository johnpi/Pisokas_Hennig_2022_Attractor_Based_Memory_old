# -*- coding: utf-8 -*-

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

# Load libraries
import numpy as np
from brian2 import *
from neurodynex.working_memory_network import wm_model

from utility_functions import *

collected_data_file = 'Data/collected_drift_trials.npy' # The file to store the collected trials data


def run_trials(num_of_trials         = 20, 
               collected_data_file   = 'Data/collected_drift_trials.npy', 
               stimulus_center_deg   = 180,
               stimulus_width_deg    = 60,
               stimulus_strength     = .06 * namp,
               t_stimulus_start      = 100 * ms,
               t_stimulus_duration   = 200 * ms,
               N_excitatory          = 1024,
               N_inhibitory          = 256,
               weight_scaling_factor = 2.0,
               sim_time_duration     = 10000. * ms
              ):
    """
        Runs trials of the activity bump drift and collects corresponding time series
        
        num_of_trials (default 20) : How many trials of the experiment to run. 
        collected_data_file (default 'Data/collected_drift_trials.npy') : File to store collected data in.
        stimulus_center_deg (default 180) : Stimulus heading. 
        
    """
    
    #num_of_trials = 20
    #collected_data_file = 'Data/collected_drift_trials.npy' # The file to store the collected trials data

    # Try to load existing data if any otherwise create an empty collection
    try:
        collected_trials_data = np.load(collected_data_file, allow_pickle=True, encoding='bytes')
    except: 
        collected_trials_data = np.array([]) # Collected trials data records list

    # stimulus_center_deg   = 180
    # stimulus_width_deg    = 60
    # stimulus_strength     = .06 * namp
    # t_stimulus_start      = 100 * ms
    # t_stimulus_duration   = 200 * ms
    #sim_time_duration     = 10000. * ms

    #N_excitatory          = 1024 # 2048
    #N_inhibitory          = 256  # 512
    #weight_scaling_factor = 2.0 # 4.0

    for iteration in xrange(num_of_trials):
        print('Trial: ', iteration+1)
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, w_profile = wm_model.simulate_wm(N_excitatory=N_excitatory, N_inhibitory=N_inhibitory, weight_scaling_factor=weight_scaling_factor, stimulus_center_deg=stimulus_center_deg, stimulus_width_deg=stimulus_width_deg, stimulus_strength=stimulus_strength, t_stimulus_start=t_stimulus_start, t_stimulus_duration=t_stimulus_duration, sim_time=sim_time_duration)

        # Calculate the population vector angle theta
        theta_ts = get_theta_time_series_vec_add(spike_monitor_excit, idx_monitored_neurons_excit, N_excitatory, t_snapshots, t_window_width)

        # Create a new dictionary with the collected data
        collected_data = dict()
        # Simulation set up info
        collected_data['stimulus_center_deg'] = stimulus_center_deg
        collected_data['stimulus_width_deg'] = stimulus_width_deg
        collected_data['t_stimulus_start'] = t_stimulus_start
        collected_data['t_stimulus_duration'] = t_stimulus_duration
        collected_data['sim_time_duration'] = sim_time_duration
        collected_data['N_excitatory'] = N_excitatory
        collected_data['N_inhibitory'] = N_inhibitory
        collected_data['weight_scaling_factor'] = weight_scaling_factor

        # Data
        #collected_data['rate_monitor_excit'] = rate_monitor_excit
        collected_data['spike_monitor_excit'] = spike_monitor_excit.spike_trains()
        #collected_data['voltage_monitor_excit'] = voltage_monitor_excit
        collected_data['idx_monitored_neurons_excit'] = idx_monitored_neurons_excit
        #collected_data['rate_monitor_inhib'] = rate_monitor_inhib
        collected_data['spike_monitor_inhib'] = spike_monitor_inhib.spike_trains()
        #collected_data['voltage_monitor_inhib'] = voltage_monitor_inhib
        collected_data['idx_monitored_neurons_inhib'] = idx_monitored_neurons_inhib
        collected_data['theta_ts'] = theta_ts
        # Add new data record to the collected trials data
        collected_trials_data = np.append(collected_trials_data, collected_data)

    # Save all data in the file
    np.save(collected_data_file, collected_trials_data, allow_pickle=True)



# Run the trials (Note this will take a long time)

# Network hyper-parameter
# [(N_excitatory, N_inhibitory, weight_scaling_factor), ...]
network_parameters = [
    (1024, 256, 2.0),
    (2048, 512, 1.0),
    (512,  128, 4.0)
]


# Collect data for three network sizes and 9 stimulus headings
for i, network_param in enumerate(network_parameters):
    for stim_degrees in xrange(0, 360+1, 45):
        print('Experiment: %3d  stim_degrees = %3d'.format(i, stim_degrees))
        run_trials(num_of_trials         = 40, 
                   collected_data_file   = collected_data_file, 
                   stimulus_center_deg   = stim_degrees,
                   N_excitatory          = network_param[0],
                   N_inhibitory          = network_param[1],
                   weight_scaling_factor = network_param[2],
                   sim_time_duration     = 10000. * ms
                  )