# Load libraries

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

import sys
import os.path

import h5py
import numpy as np
from scipy import sparse
import itertools 

from brian2 import *
from brian2tools import *
from Python_Libs.utility_functions import *

def dataset2str(obj):
    return str(u''.join([unichr(c[0]) for c in obj]))

def indicesOfNonZerosPerRow(A):
    #print(A)
    #for i in range(A.shape[0]):
    #    print(np.nonzero(A[i])[0].tolist())
    indices = [np.nonzero(A[i])[0].tolist() for i in range(A.shape[0])]
    return indices

def gen_filename(orig_filename, ext, new_filename_ext):
    """
        Replaces the last portion of the string orig_filename (from ext to the end)
        with new_filename_ext and returns the new string.
    """
    filename_init = orig_filename[:orig_filename.rfind(ext)]
    filename_result = filename_init + new_filename_ext
    return filename_result
    

def readBumpDriftExperimentData(filename):
    f2 = h5py.File(filename, 'r')
    dset = f2['cell_array']
    num_of_records = dset.shape[1]
    
    list_of_records = []
    
    for i in range(num_of_records):
        num_of_entries = dset[:,i].shape[0]
        
        exp_model                 = dataset2str(f2[dset[:,i][0]]) # String 'dros' or 'locust'
        # Vectorial components of sparse matrix containing the spikes trains
        spikes_train_sparse_data  = np.array(f2[f2[dset[:,i][1]].values()[0].name])
        spikes_train_sparse_ir    = np.array(f2[f2[dset[:,i][1]].values()[1].name])
        spikes_train_sparse_jc    = np.array(f2[f2[dset[:,i][1]].values()[2].name])
        # Combine them in a sparse array
        spikes_train              = sparse.csc_matrix( (spikes_train_sparse_data, 
                                                        spikes_train_sparse_ir, 
                                                        spikes_train_sparse_jc)) # Sparse array
        n_P_EN                    = f2[dset[:,i][2]].value[0][0]       # Number
        n_P_EG                    = f2[dset[:,i][3]].value[0][0]       # Number
        n_E_PG                    = f2[dset[:,i][4]].value[0][0]       # Number
        n_Pintr                   = f2[dset[:,i][5]].value[0][0]       # Number
        con_matrix_filename       = dataset2str(f2[dset[:,i][6]])      # string
        con_matrix_parameter_set  = f2[dset[:,i][7]].value             # np array
        inhibition_distr_type     = dataset2str(f2[dset[:,i][8]])      # string
        inhibition_width_sigma    = f2[dset[:,i][9]].value[0][0]      # Number
        dt                        = f2[dset[:,i][10]].value[0][0]     # Number
        simulation_duration       = f2[dset[:,i][11]].value[0][0]     # Number
        changed_synapse_type      = dataset2str(f2[dset[:,i][12]])    # string
        change_percent            = f2[dset[:,i][13]].value[0][0]     # Number
        con_matrix                = f2[dset[:,i][14]].value           # np array
        try:
            read_value            = f2[dset[:,i][15]].value           # np array
        except:
            read_value            = np.array([4])
        
        # We will store the stimuli in this dictionary
        stim_setups_dict = dict()
        
        if num_of_entries == 16:
            # Old data file format
            stim_neuron_list = read_value
        else:
            # New data file format
            stim_neuron_type          = dataset2str(f2[dset[:,i][15]]) # string
            inputList                 = f2[dset[:,i][16]].value        # np array
            
            # Load now all the stimuli specifications
            j = 17
            while j < num_of_entries:
                # Load the string specifyng the type of stimulus
                stim_type             = dataset2str(f2[dset[:,i][j]])  # string
                # Number of following values in this stimulus entry
                num_of_vals           = int(f2[dset[:,i][j+1]].value[0][0]) # Number
                pulseStartTime        = f2[dset[:,i][j+2]].value[0][0] # Number
                pulseDuration         = f2[dset[:,i][j+3]].value[0][0] # Number
                stim_neuron_list      = f2[dset[:,i][j+4]].value       # np array
                maxSpikeRate          = f2[dset[:,i][j+5]].value[0][0] # Number
                maxSpikeRateFactor    = f2[dset[:,i][j+6]].value[0][0] # Number
                stim_setup_entry_dict = {
                    'pulseStartTime'     : pulseStartTime,
                    'pulseDuration'      : pulseDuration,
                    'stim_neuron_list'   : stim_neuron_list, 
                    'maxSpikeRate'       : maxSpikeRate, 
                    'maxSpikeRateFactor' : maxSpikeRateFactor
                }
                # Add the stimulus to the dict of stimuli
                stim_setups_dict[stim_type] = stim_setup_entry_dict
                j = j + 2 + num_of_vals
        
        record = {
            'exp_condition'           : exp_model               , # String       : model to run (con. matrix selected)
            'spikes_train'            : spikes_train            , # Sparse array : spikes trains
            'n_P_EN'                  : n_P_EN                  , # Number of P-ENs
            'n_P_EG'                  : n_P_EG                  , # Number of P-EGs
            'n_E_PG'                  : n_E_PG                  , # Number of E-PGs
            'n_Pintr'                 : n_Pintr                 , # Number of Pintr (Delta7s)
            'con_matrix_filename'     : con_matrix_filename     , # String       : filename of connectivity matrix prototype
            'con_matrix_parameter_set': con_matrix_parameter_set, # np array     : connectivity matrix synaptic weight values
            'inhibition_distr_type'   : inhibition_distr_type   , # String       : Keyword distribution of inhibitory weights
            'inhibition_width_sigma'  : inhibition_width_sigma  , # Number       : Width of distribution of inhibitory weights
            'dt'                      : dt                      , # Number       : Simulation time step in sec
            'simulation_duration'     : simulation_duration     , # Number       : Simulation duration in sec
            'changed_synapse_type'    : changed_synapse_type    , # String       : Type of synapse modified in this experiment
            'change_percent'          : change_percent          , # Number       : Percentage of change to the synaptic strength
            'con_matrix'              : con_matrix                # np array     : The connectivity matrix
        }
        if num_of_entries == 16:
            # Old data file format
            record['stim_neuron_list'] = stim_neuron_list          # np array     : Neuron indices that were stimulated (base 1)
        else:
            # New data file format
            record['stim_neuron_type'] = stim_neuron_type
            record['inputList']        = inputList
            record['stim_setups_dict'] = stim_setups_dict
        
        list_of_records.append(record)

    f2.close()

    return list_of_records


def calcPopulationVectors(full_filename):
    """
        Reads a file with data collected using the collect_stats_long_run.m script, 
        calculates the population vector theta for each experiment data series 
        and saves the results as a list of dictionaries with the theta time series
        in a file named after the original file with a changed ending.
        Eg results for file 
        'Drift_Analysis/Data/Stats/Long_Run_Sim/converted-collect_stats_long_run_data_dros_2_1.mat' 
        will be saved in 
        'Drift_Analysis/Data/Stats/Long_Run_Sim/converted-collect_stats_long_run_data_dros_2_1_pop_vector_theta.npy' 
    """
    
    records_list = readBumpDriftExperimentData(full_filename)

    num_of_experiments = len(records_list) # Number of entries in the file

    population_vector_results_list = []    # Here we will store the results, one entry per experiment record

    for record_i in range(0, num_of_experiments):
        
        # The dimension of the spike recordings tells us ...
        num_of_neurons = records_list[record_i]['spikes_train'].shape[0] # the number of neurons in the simulation
        num_of_samples = records_list[record_i]['spikes_train'].shape[1] # the number of samples in the time series recording
        simulation_duration = records_list[record_i]['simulation_duration']    # The duration of the simulation

        dt = records_list[record_i]['dt']                     # The simulation time step
        
        # We are extracting the spike events here
        indices_per_row = indicesOfNonZerosPerRow(records_list[record_i]['spikes_train'].toarray())

        # Convert the array of spike events to two arrays, one with which neuron fired and another 
        # at which time it fired to produce a (neuron_i, timestep) list
        indices_times_tuples_list_of_lists = [zip([i] * len(elem), elem) for i,elem in enumerate(indices_per_row)]
        # Flatten to a list of tupples
        indices_times_tuples_list = list(itertools.chain.from_iterable(indices_times_tuples_list_of_lists))
        # sort by the second element (spike time)
        indices_times_tuples_list_sorted = indices_times_tuples_list.sort(key=lambda x: x[1])
        # Separate the tupples into two lists
        indices, times = zip(*indices_times_tuples_list)
        # Here we get two lists one with firing neuron number and another with time of firing event
        indices = np.array(indices)
        times = (np.array(times) * dt) * second

        # Convert firing events to SpikeMonitor object
        spikes = SpikeGeneratorGroup(num_of_neurons, indices, times)
        spike_mon = SpikeMonitor(spikes)
        net = Network(spikes, spike_mon)
        net.run(simulation_duration*second) # Need to simulate the network for populating the spike monitor

        # Calculate population vector
        t_stimulus_start = 0 * ms
        t_stimulus_duration = 500 * ms
        sim_time_duration = simulation_duration * second
        t_window_width = 200 * ms
        snapshot_interval = 100 * ms        
        # Time points to sample the firing rates
        t_snapshots = range(
            int(math.floor((t_stimulus_start+t_stimulus_duration)/ms)),  # lower bound
            int(math.floor((sim_time_duration-t_window_width/2)/ms)),  # Subtract half window. Avoids an out-of-bound error later.
            int(round(snapshot_interval/ms))  # spacing between time stamps
            )*ms
        # Neurons to consider
        N_neurons = 8 # Uses the first 8 P-EN neurons to calculate the population vector
        idx_monitored_neurons = range(0, N_neurons)
        # Calculate the population vector angle theta
        (r_ts, theta_ts) = get_theta_time_series_vec_add(spike_mon, idx_monitored_neurons, N_neurons, t_snapshots, t_window_width)
        # Create a record entry
        records_list[record_i]['theta_ts'] = theta_ts
        records_list[record_i]['t_snapshots'] = t_snapshots
        del records_list[record_i]['spikes_train'] # TO DO: Delete it for now because np.save stores it as huge full matrix
        population_vector_results_list.append({'theta_ts'   : theta_ts, 
                                               't_snapshots': t_snapshots})


    # Save all data in a file named after the source file
    full_filename_results = gen_filename(full_filename, '.mat', '_pop_vector_theta.npy')
    np.save(full_filename_results, population_vector_results_list, allow_pickle=True)

    # Also Save all data combined in a file named after the source file
    full_filename_results_2 = gen_filename(full_filename, '.mat', '_pop_vector_theta_with_metadata.npy')
    np.save(full_filename_results_2, records_list, allow_pickle=True)



def calcSpikeRates(full_filename, 
                   idx_monitored_neurons = [], # Empty list means monitor all neurons 
                   snapshot_interval = 100 * ms, 
                   t_window_width = 200 * ms, 
                   t_stimulus_start = 0 * ms, 
                   t_stimulus_duration = 500 * ms):
    """
        Reads a file with data collected using the collect_stats_long_run.m script, 
        calculates the instanteneous spike rates for each experiment data series 
        and saves the results as a list of dictionaries with the spike rates time series
        in a file named after the original file with a changed ending.
        Eg results for file 
        'Drift_Analysis/Data/Stats/Long_Run_Sim/converted-collect_stats_long_run_data_dros_2_1.mat' 
        will be saved in 
        'Drift_Analysis/Data/Stats/Long_Run_Sim/converted-collect_stats_long_run_data_dros_2_1_spike_rates.npy' 
    """
    records_list = readBumpDriftExperimentData(full_filename)


    num_of_experiments = len(records_list) # Number of entries in the file

    for record_i in range(0, num_of_experiments):
        
        print('Processing entry {}'.format(record_i))
        
        # The dimension of the spike recordings tells us ...
        num_of_neurons = records_list[record_i]['spikes_train'].shape[0] # the number of neurons in the simulation
        num_of_samples = records_list[record_i]['spikes_train'].shape[1] # the number of samples in the time series recording
        simulation_duration = records_list[record_i]['simulation_duration']    # The duration of the simulation

        dt = records_list[record_i]['dt']                     # The simulation time step
        
        # If this record has the wrong number of recorded neurons ignore it (it happens occasionally for locust).
        if num_of_neurons != (records_list[record_i]['n_P_EN'] + 
                              records_list[record_i]['n_P_EG'] + 
                              records_list[record_i]['n_E_PG'] + 
                              records_list[record_i]['n_Pintr']):
            print('           Entry {} has wrong number of neurons: Ignored'.format(record_i))
            continue
        
        # We are extracting the spike events here
        indices_per_row = indicesOfNonZerosPerRow(records_list[record_i]['spikes_train'].toarray())

        # Convert the array of spike events to two arrays, one with which neuron fired and another 
        # at which time it fired to produce a (neuron_i, timestep) list
        indices_times_tuples_list_of_lists = [zip([i] * len(elem), elem) for i,elem in enumerate(indices_per_row)]
        # Flatten to a list of tupples
        indices_times_tuples_list = list(itertools.chain.from_iterable(indices_times_tuples_list_of_lists))
        # sort by the second element (spike time)
        indices_times_tuples_list_sorted = indices_times_tuples_list.sort(key=lambda x: x[1])
        # Separate the tupples into two lists
        indices, times = zip(*indices_times_tuples_list)
        # Here we get two lists one with firing neuron number and another with time of firing event
        indices = np.array(indices)
        times = (np.array(times) * dt) * second

        # Convert firing events to SpikeMonitor object
        spikes = SpikeGeneratorGroup(num_of_neurons, indices, times)
        spike_mon = SpikeMonitor(spikes)
        net = Network(spikes, spike_mon)
        net.run(simulation_duration*second) # Need to simulate the network for populating the spike monitor

        # Calculate population vector
        sim_time_duration = simulation_duration * second
        # Time points to sample the firing rates
        t_snapshots = range(
            int(math.floor((t_stimulus_start+t_stimulus_duration)/ms)),  # lower bound
            int(math.floor((sim_time_duration-t_window_width/2)/ms)),  # Subtract half window. Avoids an out-of-bound error later.
            int(round(snapshot_interval/ms))  # spacing between time stamps
            )*ms

        # Neurons to consider
        if len(idx_monitored_neurons) == 0:
            idxs_of_monitored_neurons = range(0, num_of_neurons)
        else:
            idxs_of_monitored_neurons = idx_monitored_neurons

        # Calculate the population vector angle theta
        spike_rates_ts = get_spike_rates(spike_mon, idxs_of_monitored_neurons, t_snapshots, t_window_width)
        # Create a record entry
        records_list[record_i]['spike_rates_ts'] = spike_rates_ts
        records_list[record_i]['t_snapshots'] = t_snapshots
        del records_list[record_i]['spikes_train'] # TO DO: Delete it for now because np.save stores it as huge full matrix


    # Save all data combined in a file named after the source file
    full_filename_results = gen_filename(full_filename, '.mat', '_spike_rates.npy')
    np.save(full_filename_results, records_list, allow_pickle=True)

