# -*- coding: utf-8 -*-

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

# Load libraries
import sys
import argparse # For command line argument parsing

import numpy as np
from brian2 import *
from neurodynex.working_memory_network import wm_model
from neurodynex.working_memory_network import wm_model_modified

from utility_functions import *

collected_data_file = 'Data/collected_drift_trials_v2.npy' # The file to store the collected trials data


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
               sim_time_duration     = 10000. * ms,
               t_window_width      = 200*ms,
               snapshot_interval   = 100*ms,
               synaptic_noise_amount = 0.0
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

    #t_window_width      = 200*ms
    #snapshot_interval   = 100*ms


    for iteration in range(num_of_trials):
        print('Trial: ', iteration+1)
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, w_profile = wm_model_modified.simulate_wm(N_excitatory=N_excitatory, N_inhibitory=N_inhibitory, weight_scaling_factor=weight_scaling_factor, stimulus_center_deg=stimulus_center_deg, stimulus_width_deg=stimulus_width_deg, stimulus_strength=stimulus_strength, t_stimulus_start=t_stimulus_start, t_stimulus_duration=t_stimulus_duration, sim_time=sim_time_duration, synaptic_noise_amount = synaptic_noise_amount)
        
        t_snapshots = range(
            int(math.floor((t_stimulus_start+t_stimulus_duration)/ms)),  # lower bound
            int(math.floor((sim_time_duration-t_window_width/2)/ms)),  # Subtract half window. Avoids an out-of-bound error later.
            int(round(snapshot_interval/ms))  # spacing between time stamps
            )*ms
        
        # Calculate the population vector angle theta
        theta_ts = get_theta_time_series_vec_add(spike_monitor_excit, 
                                                 idx_monitored_neurons_excit, 
                                                 len(idx_monitored_neurons_excit), # Instead of N_excitatory because
                                                                                   # wm_model_modified.simulate_wm returns 
                                                                                   # in idx_monitored_neurons_excit at most 
                                                                                   # 1024 neurons 
                                                 t_snapshots, 
                                                 t_window_width)

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
        collected_data['synaptic_noise_amount'] = synaptic_noise_amount

        # Data
        #collected_data['rate_monitor_excit'] = rate_monitor_excit
        collected_data['spike_monitor_excit'] = spike_monitor_excit.spike_trains()
        #collected_data['voltage_monitor_excit'] = voltage_monitor_excit
        collected_data['idx_monitored_neurons_excit'] = idx_monitored_neurons_excit
        #collected_data['rate_monitor_inhib'] = rate_monitor_inhib
        collected_data['spike_monitor_inhib'] = spike_monitor_inhib.spike_trains()
        #collected_data['voltage_monitor_inhib'] = voltage_monitor_inhib
        collected_data['idx_monitored_neurons_inhib'] = idx_monitored_neurons_inhib
        
        collected_data['t_window_width']    = t_window_width
        collected_data['snapshot_interval'] = snapshot_interval
        collected_data['t_snapshots']       = t_snapshots
        collected_data['theta_ts']          = theta_ts
        # Add new data record to the collected trials data
        collected_trials_data = np.append(collected_trials_data, collected_data)

    # Save all data in the file
    np.save(collected_data_file, collected_trials_data, allow_pickle=True)



# Run the trials (Note this will take a long time)

# Network hyper-parameter
# [(N_excitatory, N_inhibitory, weight_scaling_factor), ...]
# Obsolete

network_parameters = [
    (512,  128, 4.0),
    (1024, 256, 2.0),
    (2048, 512, 1.0),
    (4096, 1024, 0.5),
    (8192, 2048, 0.25)
]
network_parameters_dict = {
      64: (64,    16, 32.0),
     128: (128,   32, 16.0),
     256: (256,   64,  8.0),
     512: (512,  128,  4.0),
    1024: (1024, 256,  2.0),
    2048: (2048, 512,  1.0),
    4096: (4096, 1024, 0.5),
    8192: (8192, 2048, 0.25),
   16384: (16384,4096, 0.125),
   32768: (32768,8192, 0.0625)
}


# Obsolete

synaptic_noise_amount_list = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]


# Obsolete
# Collect data for three network sizes and 9 stimulus headings
def explore_heading_angles():
    for i, network_param in enumerate(network_parameters):
        for stim_degrees in range(0, 360, 45):
            print('Experiment: {:3}  stim_degrees = {:3}'.format(i+1, stim_degrees))
            run_trials(num_of_trials         = 1, 
                       collected_data_file   = collected_data_file, 
                       stimulus_center_deg   = stim_degrees,
                       N_excitatory          = network_param[0],
                       N_inhibitory          = network_param[1],
                       weight_scaling_factor = network_param[2],
                       sim_time_duration     = 10000. * ms,
                       synaptic_noise_amount = 0.0
                      )

# Obsolete
# Collect data for three network sizes and different noise levels
def explore_noise_levels():
    for i, network_param in enumerate(network_parameters):
        for synaptic_noise_amount in synaptic_noise_amount_list:
            print('Experiment: {:3}  synaptic_noise_amount = {:3}'.format(i+1, synaptic_noise_amount))
            run_trials(num_of_trials         = 10, 
                       collected_data_file   = collected_data_file, 
                       stimulus_center_deg   = 180,
                       N_excitatory          = network_param[0],
                       N_inhibitory          = network_param[1],
                       weight_scaling_factor = network_param[2],
                       sim_time_duration     = 10000. * ms,
                       synaptic_noise_amount = synaptic_noise_amount
                      )

# Collect data for different network sizes, noise levels, and stimulus headings
def explore_spec_setups(N_excitatory_neurons_list, synaptic_noise_amount_list, stim_heading_degrees_list, N_trials, filename):
    for i, Ne in enumerate(N_excitatory_neurons_list):
        for stim_heading_degrees in stim_heading_degrees_list:
            for synaptic_noise_amount in synaptic_noise_amount_list:
                for trial in range(1, N_trials+1):
                    print('Trial {:3} with {:5} neurons, stimulus at {:3}deg, synaptic noise {:3}SNR.'.format(
                            trial, 
                            Ne, 
                            stim_heading_degrees, 
                            synaptic_noise_amount))
                    network_param = network_parameters_dict.get(Ne)
                    if network_param is None:
                        print('ERROR: Unlisted number of excitatory neurons: {}'.format(Ne))
                    else:
                        run_trials(num_of_trials         = 1, 
                                   collected_data_file   = filename, 
                                   stimulus_center_deg   = stim_heading_degrees,
                                   N_excitatory          = network_param[0],
                                   N_inhibitory          = network_param[1],
                                   weight_scaling_factor = network_param[2],
                                   sim_time_duration     = 10000. * ms,
                                   synaptic_noise_amount = synaptic_noise_amount
                                  )


    

# Main program

#if len(sys.argv) < 2 or str(sys.argv[0]) == 'explore=everything':
#    explore_heading_angles()
#    explore_noise_levels()
#elif str(sys.argv[1]) == 'explore=angles':
#    explore_heading_angles()
#elif str(sys.argv[1]) == 'explore=noise':
#    explore_noise_levels()
#else:
#    print('Error: Unknown command line argument given. ARG: {}'.format(sys.argv))


# New command line options set up
parser = argparse.ArgumentParser(description='Run the ring attractor simulation and collect data.')
# Expect the number of excitatory neurons to use
parser.add_argument('-N', '--neurons_num_exc', type=int, nargs='+', dest='neurons_num_exc', required=True, 
                   help='One or more integers specifying the number of excitatory neurons in the network. Simulations will be run for each of these sizes.')
# Expect the amount of synaptic noise to use
parser.add_argument('--weight_noise_SNR', type=float, nargs='+', dest='weight_noise_SNR', default=[0], 
                   help='One or more real numbers specifying the amount of synaptic weight noise (as SNR ratio) to use.')
# Expect the stimulus heading to use
parser.add_argument('--heading', type=int, nargs='+', dest='headings', default=[180], 
                   help='One or more integer numbers specifying the stimulus heading to use.')
# How many trials to run for each condition
parser.add_argument('-t', '--trials', type=int, dest='trials', default=20, 
                   help='Number of simulations to run and collect results. Default 20 trials.')
# File to append collected results to
parser.add_argument('-f', '--file', type=str, dest='filename', required=True, 
                   help='Filename to append collected results to.')

# Parse the command line arguments
args = parser.parse_args()

N_neurons_exc_list = args.neurons_num_exc
weight_noise_SNR_list = args.weight_noise_SNR
headings_list = args.headings
N_trials = args.trials
filename = args.filename

# Run trials
explore_spec_setups(N_excitatory_neurons_list=N_neurons_exc_list, 
                    synaptic_noise_amount_list=weight_noise_SNR_list, 
                    stim_heading_degrees_list = headings_list, 
                    N_trials = N_trials,
                    filename = filename)
