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

