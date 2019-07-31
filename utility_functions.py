# -*- coding: utf-8 -*-

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

# Load libraries
import numpy as np
from brian2 import *


# Decoding the population activity into a population vector

def get_orientation(idx_list, N):
    """
        idx_list : list indeces of neurons
        N        : Number of neurons
        returns  : list of theta angle preferences of neurons in the list of indeces
        
        The circle 360 is broken into N intervals and the angle corresponding in 
        the middle of the interval is calculated. 
    """
    interval = 360. / N 
    interval_middle  = 360. / N / 2

    return  (interval_middle + interval * np.array(idx_list)).tolist()


def get_spike_count(spike_monitor, spike_index_list, t_min, t_max):
    """
        Returns : A list of spike counts
    """
    nr_neurons = len(spike_index_list)
    spike_count_list = np.zeros(nr_neurons)
    if isinstance(spike_monitor, SpikeMonitor):
        spike_trains = spike_monitor.spike_trains()
    elif isinstance(spike_monitor, dict): # If it is a dict() assume it is the result of spike_trains()
        spike_trains = spike_monitor
    
    spike_count_list = []
    
    for idx in spike_index_list:
        num_of_spikes = np.sum(np.logical_and(spike_trains[idx] >= t_min, spike_trains[idx] < t_max))
        spike_count_list.append(num_of_spikes)
    
    return spike_count_list

def add_vectors(vectors_list, polar_or_cartesian='polar', angles_in='rads'):
    """
        Adds a list of vectors and returns the total vector. Receives a list of 
        vector tupples or lists in polar or cartesian coordinates and returns 
        the vector addition of all of them. 
        
        vectors_list : a list of vector tupples or lists in polar or cartesian coordinates.
        polar_or_cartesian : 'polar' or 'cartesian' specifies the coordinate system used.
        angles_in          : 'rads' or 'degrees' specifies how the angular values should be interpreted.
        Returns : A tupple (x_total, y_total, r_total, theta_total)
    """
    cartesian_vector_list = []
    if polar_or_cartesian=='polar':
        for v in vectors_list:
            r, th = v
            if angles_in == 'degrees':
                th = np.radians(th)
            if th > (2*np.pi):
                th = th % (2*np.pi)
            x = r * np.cos(th)
            y = r * np.sin(th)
            cartesian_vector_list.append([x, y])
        
    if polar_or_cartesian=='cartesian':
        cartesian_vector_list = vectors_list
    
    # Calculate total of vector addition
    x_total = 0.
    y_total = 0.
    for v in cartesian_vector_list:
        x, y = v
        x_total += x
        y_total += y

    # Convert to polar coordinates
    r_total     = np.sqrt(x_total**2 + y_total**2)
    #theta_total = np.arctan(y_total / x_total)
    theta_total = np.arctan2(y_total, x_total)
    
    # arctan2 returns a mapping of 0 to 180 to the range 0 to 180 and 181 to 360 to the range -179 to -1
    if theta_total < 0.:
        theta_total = (2*np.pi) + theta_total
    
    if theta_total > (2*np.pi):
        theta_total = theta_total % (2*np.pi)

    if angles_in == 'degrees':
        theta_total = np.degrees(theta_total)
    
    return (x_total, y_total, r_total, theta_total)


def get_theta_time_series_vec_add(spike_monitor, idx_monitored_neurons, total_num_of_neurons, t_snapshots, t_window_width):
    """ 
        This implementation is more accurate. It uses the add_vectors() function to 
        add the constituent vectors in order to derive the population coded vector. 
    """
    theta_angles = get_orientation(idx_monitored_neurons, total_num_of_neurons)
    theta_angles = np.array(theta_angles)
    theta_time_series = []
    for t_snap in t_snapshots:
        t_min = t_snap - t_window_width/2
        t_max = t_snap + t_window_width/2
        spike_counts = get_spike_count(spike_monitor, idx_monitored_neurons, t_min, t_max)
        spike_counts_ndarray = np.array(spike_counts)
        
        polar_vectors_list = list(zip(spike_counts_ndarray, theta_angles))
        x_total, y_total, r_total, theta_total = add_vectors(polar_vectors_list, 
                                                             polar_or_cartesian='polar', 
                                                             angles_in='degrees')
        
        theta_time_series.append(theta_total)

    return theta_time_series

def get_theta_time_series(spike_monitor, idx_monitored_neurons, total_num_of_neurons, t_snapshots, t_window_width):
    """
        This implementation is the typical population activity weighted vector. It is 
        inaccurate near 0deg as well as when there is noise in the population activity. 
    """
    theta_angles = get_orientation(idx_monitored_neurons, total_num_of_neurons)
    theta_angles = np.array(theta_angles)
    theta_time_series = []
    for t_snap in t_snapshots:
        t_min = t_snap - t_window_width/2
        t_max = t_snap + t_window_width/2
        spike_counts = get_spike_count(spike_monitor, idx_monitored_neurons, t_min, t_max)
        spike_counts_ndarray = np.array(spike_counts)
        
        theta_time_series.append(np.sum(spike_counts_ndarray * theta_angles) / np.sum(spike_counts_ndarray))

    return theta_time_series



# Functions for selecting records matching conditions from list. 

def check_value(record, key, expected_value):
    """
        Returns True if the dictionary record contains a key 
        named 'key' with value equal to expected_value.
    """
    if key in record and record[key] == expected_value:
        return True
    return False

def select_records_from_list(dicts_list, selectors, operator):
    """
        Gets a list of dicts and a dict of selector field-value pairs 
        and a logical operator 'or' or 'and'. Returns a list of only 
        the dict entries that satisfy the conditions in selectors based
        on the operator.
    """
    selected_data_items = []
    track_selected_items_indxs = []   # Keeps track of which items in the list have already been selected to avoid double entries
    
    # Iterate over all records in the loaded data file and select those 
    # matching *at least one* of the specified expected values. That is, 
    # a selection using an OR operator across the specified values. 
    for i, item in enumerate(dicts_list):
        # Iterate over function arguments
        for arg_key, arg_value in list(selectors.items()):
            if arg_value is not None:
                if check_value(item, key=arg_key, expected_value=arg_value):
                    if not i in track_selected_items_indxs:
                        selected_data_items.append(item)
                        track_selected_items_indxs.append(i)

    # Now remove the records that do not satisfy all specified values. 
    # That is, an AND operation.
    items_to_remove_list = []
    if operator == 'and':
        for i, item in enumerate(selected_data_items):
            # Iterate over function arguments
            for arg_key, arg_value in list(selectors.items()):
                if arg_value is not None:
                    if not check_value(item, key=arg_key, expected_value=arg_value):
                        items_to_remove_list.append(i) # indices of items to remove
    # remove the records
    selected_data_items_and_op = [selected_data_items[i] for i, elem in enumerate(selected_data_items) if i not in items_to_remove_list]
    selected_data_items = selected_data_items_and_op
    
    return selected_data_items


# Example usage:
#selected_data_items = pick_data_samples('Data/collected_drift_trials_v2_angles.npy', 
#                      stimulus_center_deg   = 180,
#                      stimulus_width_deg    = None,
#                      sim_time_duration     = None,
#                      N_excitatory          = 1024,
#                      synaptic_noise_amount = None, 
#                      operator              = 'or'  # Records with at least one matching value are returned
#                     )
def pick_data_samples(collected_data_file, 
                      stimulus_center_deg   = None,
                      stimulus_width_deg    = None,
                      sim_time_duration     = None,
                      N_excitatory          = None,
                      synaptic_noise_amount = None, 
                      operator              = 'or'  # Records with at least one matching value are returned
                     ):
    
    """
        From a collection of records stored in a list or np.array it selects 
        only those satisfying the specified conditions. The conditions are 
        values that keys in each record must have. Only keys with not None 
        value are considered. The conditions are checked in an OR fashion if 
        operator is 'or' but in and AND fashion if operator is 'and'. 
        It returns a list containing only the matching records. 
        
        collected_data_file     : The data file to read records list from.
        operator (default 'or') : If 'or' records with at least one matching value are returned.
                                  If 'and' only records matching all values are returned.
        All other arguments are the conditions to check. The name of each argument variable is 
        used as the key and its value as the constraint. 
    """
    
    args = locals() # Get function arguments as dict()
    args['collected_data_file'] = None # Set to None so that it is ignored in the iterator later
    args['operator']            = None # Set to None so that it is ignored in the iterator later
    
    # Try to load existing data if any otherwise create an empty collection
    try:
        collected_trials_data = np.load(collected_data_file, allow_pickle=True, encoding='bytes')
    except: 
        collected_trials_data = np.array([]) # Collected trials data records list

    selected_data_items = select_records_from_list(collected_trials_data, args, operator)
    
    return selected_data_items



def mean_absolute_deviation(expected_value, time_series_list):
    """
        Returns the mean absolute deviation (MAD) of time series produced 
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
        Returns          : A time series with elements the MAD at each point in 
                           time. It has the same number of items as the contained 
                           time series. 
    """
    
    # The absolute value of the differences of the time series samples from the expected value
    
    abs_diff_list = []
    # Get the absolute deviation of each item of the series from the expected value
    for ts in time_series_list:
        abs_diff = np.abs(np.ones(len(ts)) * expected_value - ts)
        abs_diff_list.append(abs_diff)
    
    # Get the mean absolute deviation across all time series at each point in time
    abs_diff_mean = np.mean(abs_diff_list, axis=0)
    
    return abs_diff_mean






# Experimental and untested

def add_gaussian_white_noise_by_variance(data, variance):
    """
        Adds Additive Gaussian White Noise to the given data.
        data : an numpy array with the data to noisify.
        variance : the gaussian noise variance. 
        Since the average power of a random variable $X$ is $E[X^2] = \mu^2 + \sigma^2$ 
        and we set $\mu = 0$, it follows that $E[X^2] = \sigma^2$ thus the power of the 
        variable is determined by its variance. 
        Returns : the data with added noise.
    """
    sigma_sd = np.sqrt(variance)
    noise = np.random.normal(loc=0., scale=sigma_sd, size=np.array(data).shape)
    noisy_data = np.array(data) + noise
    if isinstance(data, list):
        noisy_data = noisy_data.tolist()
    return noisy_data

def add_gaussian_white_noise_by_magnitude(data, noise_portion):
    """
        Details on the computation at 
            https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
            https://uk.mathworks.com/matlabcentral/answers/40772-snr-in-awgn
            https://dsp.stackexchange.com/questions/33849/adding-awgn-noise-with-a-correct-noise-power-to-the-signal
            
            https://www.tcd.ie/Physics/research/groups/magnetism/files/lectures/py5021/MagneticSensors3.pdf
            http://www.commsp.ee.ic.ac.uk/~cling/Com/Problems.pdf
            http://web.mit.edu/6.02/www/f2010/handouts/lectures/L4-notes.pdf
            https://web.stanford.edu/group/cioffi/doc/book/chap1.pdf
            https://dsp.stackexchange.com/questions/30822/additive-gaussian-white-noise-bandwidth
            https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function/
        
        Adds Additive Gaussian White Noise to the given data. Assumes that the data in the 
        array 'data' are representative of the possible values in the population for 
        calculating the power of the signal. 
        data            : an numpy array with the data to noisify.
        noise_magnitude : the amount of noise to add. Let that be a portion of noise in (0,1)
        Returns         : the data with added noise.
    """
    signal_amplitude = np.max(np.max(data)) # - np.min(np.min(data))
    signal_power = signal_amplitude ** 2
    signal_power_dB = 10. * np.log10(signal_power)
    
    signal_av_power = np.mean(signal_power)
    signal_av_power_dB = 10. * np.log10(signal_av_power)
    target_noise_av_power = noise_portion * signal_av_power
    target_noise_av_power_dB = 10. * np.log10(target_noise_av_power) # This results to -inf for target_noise_av_power=0. Dont worry about the warning that there was division by 0 in log10.
    #target_noise_av_power_dB = 10. * np.log10(np.max([target_noise_av_power, 10**-100])) # To avoid division by 0 (not really needed)
    #target_SNR = signal_av_power / target_noise_av_power # Danger of division by 0
    #target_SNR_dB = 10. * np.log10(target_SNR)
    target_SNR_dB = signal_av_power_dB - target_noise_av_power_dB
    
    noise_av_dB = signal_av_power_dB - target_SNR_dB
    noise_av_power = 10. ** (noise_av_dB / 10.)
    
    variance = noise_av_power
    
    noisy_data = add_gaussian_white_noise_by_variance(data, variance)
    return noisy_data



# Not perfect but works well enough
def unwrap_modulo_time_series(data_time_series, modulo = 360):
    """
        Gets a time series with values wrapped around a modulo value, 
        such as heading angles wrapped around a 360 degree circle, and 
        returns an unwrapped time series. It assumes that heading angle 
        values move around the circle continuously so jumps from near 0 
        to near 360 degrees and vice versa are intrepreted as incremental 
        angle changes. 
        Tested with additive gaussian white noise and can tolerate up 
        to 1/SNR = 2*10^-7
    """
    modulo_range = [0, modulo]
    dp_mod = data_time_series

    dp_step_diffs  = np.array([])
    dp_reconstruct = np.array([])

    for i,d in enumerate(dp_mod):
        if i == 0:
            init_value = dp_mod[i]
            dp_step_diffs = np.append(dp_step_diffs, 0)
        else:
            step_diff = dp_mod[i] - dp_mod[i-1]
            dp_step_diffs = np.append(dp_step_diffs, step_diff)

    for i,d in enumerate(dp_mod):
        if i == 0:
            dp_reconstruct = np.append(dp_reconstruct, init_value)
        elif np.abs(dp_step_diffs[i]) < modulo_range[1] / 2:
            dp_reconstruct = np.append(dp_reconstruct, dp_reconstruct[-1] + dp_step_diffs[i])
        else:
            if dp_step_diffs[i] < 0: # The curve was going increasing value
                dp_reconstruct = np.append(dp_reconstruct, dp_reconstruct[-1] + dp_step_diffs[i] + modulo_range[1])
            else:                    # The curve was going decreasing value
                dp_reconstruct = np.append(dp_reconstruct, dp_reconstruct[-1] + dp_step_diffs[i] - modulo_range[1])
    
    # Return recorstructed time series
    return dp_reconstruct
