# Optimisation of synaptic conductances 

# Load libraries

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

import fire 

import math
import numpy as np
from brian2 import *

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from neurodynex.working_memory_network import wm_model_modified_simplified_EC_LV_principal

from neurodynex.tools import plot_tools

from utility_functions import *

import pandas as pd
from scipy.optimize import curve_fit # for doing regression
from sklearn.metrics import r2_score # for measuring fit error
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import dual_annealing

from skopt import gp_minimize
from skopt.space import Real
#from skopt.plots import plot_convergence

from Python_Libs.utility_functions import *

# Objective function parameters
W_desired =  55.0       # Desired bump width
W_desired_inhib = 360.0 # Desired width of inhibitory neurons activity (all active)
H_desired = 180.0       # Desired bump angle
A_desired = 10          # Desired bump min amplitude


model_opt        = 'EC_LV' # EC_LV or simple
N_excitatory_opt = 512
N_inhibitory_opt = 128
poisson_firing_rate_opt = 0
G_extern2excit_opt = 24
stimulus_strength_opt = 2

# Useful for the optimisation of synaptic conductances 
def get_full_width_at_half_maximum_modified(values_array, circular=False, value_range=None):
    peak_idx = np.argmax(values_array) # get the index of the first max value
    max_v = np.max(values_array)       # max value in the array
    min_v = np.min(values_array)       # min value in the array
    peak_v = values_array[peak_idx]    # signal value at the peak
    half_v = (max_v - min_v) / 2       # half of the peak to peak value of the signal
    
    # We want to find the x axis width
    # Shift signal values so that at half maximum the values are 0
    values_array_shifted = values_array - half_v
    # Fit a function to the data damples in order to precisely find 0 crossing points along the x axis
    fitted_curve = interpolate.UnivariateSpline(range(len(values_array)), values_array_shifted, s=0)
    roots = fitted_curve.roots() # get the x axis points where the function crosses y=0
    # Get the nearest y=0 crossing points at the left and right of the peak
    diffs = roots - peak_idx
    
    # If the array represents elements on a circle
    if circular:
        # If only two 0 crossings and both on the same side of the peak try to correct for wrap
        if  len(diffs) == 2 and diffs[0] * diffs[1] > 0:
            max_i = np.argmax(np.abs(diffs)) # Which of the two differences is furthest away from the peak
            if max_i == 0: # In this case both 0 crossings are on the left of the peak
                diffs[max_i] =   len(values_array)-1 - peak_idx + roots[0] # Wrap the leftmost 0 crossing
            if max_i == 1: # In this case both 0 crossings are on the right of the peak
                diffs[max_i] = -(len(values_array)-1 + peak_idx - roots[1]) # Wrap the rightmost 0 crossing
        # This is an unrealistic situation where the peak is on the one end of the array only
        if len(diffs) == 1:
            diffs = np.append(diffs, -diffs[0]) # Assume symmetry
    
    # Modified these from the library version because it was not working for noisy bump. 
    # It was selecting the nearest 0 crossings now selects the farthest.
    #nearest_crossing_on_left = np.max(diffs[diffs<=0], initial=-inf)
    #nearest_crossing_on_right = np.min(diffs[diffs>=0], initial=inf)
    nearest_crossing_on_left = np.min(diffs[diffs<=0], initial=inf)
    nearest_crossing_on_right = np.max(diffs[diffs>=0], initial=-inf)
    # The distances of nearest crossing around the peak give us the FWHM
    FWHM = nearest_crossing_on_right - nearest_crossing_on_left
    
    # Convert to a number scaled to the range of values
    if value_range is not None:
        FWHM = FWHM / len(values_array) * value_range
    
    return FWHM


def get_full_width_at_half_maximum_ts_modified(values_array_of_arrays, circular=False, value_range=None):
    FWHM_list = []
    for i, values_array in enumerate(values_array_of_arrays):
        FWHM = get_full_width_at_half_maximum_modified(values_array, circular, value_range)
        FWHM_list.append(FWHM)
    
    return np.array(FWHM_list)



# Calculate spike rates
def calcSpikeRates(spike_mon,
                   N_neurons, # Number of neurons to calculate the population vector of
                   t_stimulus_start = 0 * ms, 
                   t_stimulus_duration = 500 * ms, 
                   sim_time_duration = 500. * ms, 
                   t_window_width = 200 * ms, 
                   snapshot_interval = 100 * ms):
    # Time points to sample the firing rates
    t_snapshots = range(
        int(math.floor((t_stimulus_start+t_stimulus_duration)/ms)),  # lower bound
        int(math.floor((sim_time_duration-t_window_width/2)/ms)),  # Subtract half window. Avoids an out-of-bound error later.
        int(round(snapshot_interval/ms))  # spacing between time stamps
        )*ms
    # Neurons to consider
    idx_monitored_neurons = range(0, N_neurons)
    # Calculate the population vector angle theta
    spike_rates_ts = get_spike_rates(spike_mon, idx_monitored_neurons, t_snapshots, t_window_width)
    # Create a record entry
    return (t_snapshots, spike_rates_ts)
    
    
# Calculate population vector
def calcPopulationTheta(
    spike_mon, 
    N_neurons, # Number of neurons to calculate the population vector of
    t_stimulus_start = 0 * ms, 
    t_stimulus_duration = 500 * ms, 
    sim_time_duration = 500. * ms, 
    t_window_width = 200 * ms, 
    snapshot_interval = 100 * ms):
    # Time points to sample the firing rates
    t_snapshots = range(
        int(math.floor((t_stimulus_start+t_stimulus_duration)/ms)),  # lower bound
        int(math.floor((sim_time_duration-t_window_width/2)/ms)),  # Subtract half window. Avoids an out-of-bound error later.
        int(round(snapshot_interval/ms))  # spacing between time stamps
        )*ms
    # Neurons to consider
    idx_monitored_neurons = range(0, N_neurons)
    # Calculate the population vector angle theta
    (r_ts, theta_ts) = get_theta_time_series_vec_add(spike_mon, idx_monitored_neurons, N_neurons, t_snapshots, t_window_width)
    theta_ts = np.array(theta_ts)
    return (t_snapshots, r_ts, theta_ts)

def calc_FWHM(spike_rates_ts, erroneous_samples_value=0, circular=False, value_range=None):
    FWHM_list = get_full_width_at_half_maximum_ts_modified(spike_rates_ts, circular=circular, value_range=value_range)
    if erroneous_samples_value is not None:
        FWHM_list[FWHM_list==inf] = erroneous_samples_value
        FWHM_list[FWHM_list==-inf] = erroneous_samples_value
    return FWHM_list
    

    
# Optimisation of synaptic conductances 
from scipy.optimize import minimize
from scipy.optimize import basinhopping

def calc_objective_value(
    spike_monitor_excit, N_excitatory, 
    spike_monitor_inhib, N_inhibitory, 
    t_stimulus_start = 100*ms, 
    t_stimulus_duration = 200*ms, 
    sim_time_duration = 500. * ms,
    t_window_width = 20 * ms, 
    snapshot_interval = 10 * ms, 
    W_desired =  60,       # Desired bump width
    W_desired_inhib = 360, # Desired width of inhibitory neurons activity (all active)
    H_desired = 180,       # Desired bump angle
    A_desired = 10         # Desired bump min amplitude
    ):
    # Calculate profile of excitatory neurons
    (t_snapshots, spike_rates_ts) = calcSpikeRates(
        spike_monitor_excit, N_excitatory, 
        #spike_monitor_inhib, N_inhibitory, 
        t_stimulus_start = t_stimulus_start, 
        t_stimulus_duration = t_stimulus_duration, 
        sim_time_duration = sim_time_duration,
        t_window_width = t_window_width, 
        snapshot_interval = snapshot_interval)

    FWHM_list = calc_FWHM(spike_rates_ts.T, erroneous_samples_value=360, circular=True, value_range=360)

    (t_snapshots, r_ts, theta_ts) = calcPopulationTheta(
        spike_monitor_excit, N_excitatory, 
        #spike_monitor_inhib, N_inhibitory, 
        t_stimulus_start = t_stimulus_start, 
        t_stimulus_duration = t_stimulus_duration, 
        sim_time_duration = sim_time_duration,
        t_window_width = t_window_width, 
        snapshot_interval = snapshot_interval)

    # Calculate profile of inhibitory neurons
    (t_snapshots_inhib, spike_rates_ts_inhib) = calcSpikeRates(
        #spike_monitor_excit, N_excitatory, 
        spike_monitor_inhib, N_inhibitory, 
        t_stimulus_start = t_stimulus_start, 
        t_stimulus_duration = t_stimulus_duration, 
        sim_time_duration = sim_time_duration,
        t_window_width = t_window_width, 
        snapshot_interval = snapshot_interval)

    FWHM_list_inhib = calc_FWHM(spike_rates_ts_inhib.T, erroneous_samples_value=0, circular=True, value_range=360)

    # Measure the bump characteristics after the stimulus
    spike_rates_mean = np.mean(spike_rates_ts/Hz, axis=1)
    peak2peak = np.max(spike_rates_mean) - np.min(spike_rates_mean)
    e_W_all = np.abs(W_desired - np.mean(FWHM_list))/ 360
    e_W_all_inhib = np.abs(W_desired_inhib - np.mean(FWHM_list_inhib))/ 360
    e_H_all = np.abs(H_desired - np.mean(theta_ts)) / 360
    #e_A_all = (int(peak2peak < A_desired) * np.abs(A_desired+0.1 - peak2peak))/10.0
    e_A_all = 1. / (1. + peak2peak)
    
    spike_rates_mean_all_inhib = np.mean(spike_rates_ts_inhib)
    e_S_all_inhib = 1. / (1. + spike_rates_mean_all_inhib/Hz)
    #e_S_all_inhib_2 = 1. / (1. + sum((np.mean(spike_rates_ts_inhib, axis=0)/Hz) > 30)/spike_rates_ts_inhib.shape[1])
    e_S_all_inhib_2 = 2*(1. / (1. + sum((np.mean(spike_rates_ts_inhib, axis=0)/Hz) > 30)/spike_rates_ts_inhib.shape[1]))-1
    
    # Measure the bump characteristics two time snapshot windows before the last
    t_snapshots[len(t_snapshots)-3]
    spike_rates_mean_2 = (spike_rates_ts/Hz).T[len(t_snapshots)-3]
    peak2peak_2 = np.max(spike_rates_mean_2) - np.min(spike_rates_mean_2)
    e_W_2 = np.abs(W_desired - FWHM_list[len(t_snapshots)-3])/ 360
    e_H_2 = np.abs(H_desired - theta_ts[len(t_snapshots)-3]) / 360
    e_A_2 = 1. / (1. + peak2peak_2)
    
    # Measure how much does the bump width change
    W_diff = (np.mean(FWHM_list[len(t_snapshots)-50:len(t_snapshots)-40]) - np.mean(FWHM_list[len(t_snapshots)-213:len(t_snapshots)-203])) / 360
    
    #fitness_error = 4*e_W_all + e_H_all + e_A_all
    #fitness_error = 5 * (e_W_all + e_W_2 + e_W_all_inhib) + e_H_all + e_H_2 + e_A_all + e_A_2 + e_S_all_inhib_2*2
    fitness_error = 5 * (e_W_all + e_W_2) + e_H_all + e_H_2 + e_A_all + e_A_2 + e_S_all_inhib_2*2 + W_diff
    return fitness_error


def objective_function(x): 
    #sim_time_duration    = 500. * ms
    sim_time_duration    = 10000. * ms
    t_stimulus_start     = 100*ms
    t_stimulus_duration  = 200*ms
    stimulus_center_deg  = 180
    stimulus_width_deg   = 40
    N_excitatory         = N_excitatory_opt
    N_inhibitory         = N_inhibitory_opt
    weight_scaling_factor= 1.
    sigma_weight_profile = 20.
    #stimulus_strength    = 0.5 * namp
    #stimulus_strength    = 2.0 * namp
    #poisson_firing_rate  = 2.3 * Hz
    #poisson_firing_rate  = 0.0 * Hz
    poisson_firing_rate  = poisson_firing_rate_opt * Hz
    G_extern2excit       = G_extern2excit_opt * nS
    stimulus_strength    = stimulus_strength_opt * namp
    Jpos_excit2excit     = 1.6

    G_inhib2inhib        = x[0] * nS  # 0.3584 * nS #* 0.
    G_inhib2excit        = x[1] * nS  # 0.4676 * nS #* 4.
    G_excit2excit        = x[2] * nS  # 0.13335 * nS# / 2.
    G_excit2inhib        = x[3] * nS  # 0.12264 * nS * 98 # = 12nS
    if len(x) == 5:
        g_coop           = x[4] * nS # Initial 400pS
    else:
        g_coop           = 0.0 * nS # Initial 400pS
    
    # Actual model with mAChR channels
    if model_opt == 'EC_LV':
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, weight_profile_45 = wm_model_modified_simplified_EC_LV_principal.simulate_wm_EC_LV_Principal_Neurons(sim_time=sim_time_duration, poisson_firing_rate=poisson_firing_rate, sigma_weight_profile=sigma_weight_profile, Jpos_excit2excit=Jpos_excit2excit, t_stimulus_start=t_stimulus_start, t_stimulus_duration=t_stimulus_duration, stimulus_center_deg=stimulus_center_deg,stimulus_width_deg=stimulus_width_deg,N_excitatory=N_excitatory,N_inhibitory=N_inhibitory,weight_scaling_factor=weight_scaling_factor,stimulus_strength=stimulus_strength, G_inhib2inhib=G_inhib2inhib, G_inhib2excit=G_inhib2excit, G_excit2excit=G_excit2excit, G_excit2inhib=G_excit2inhib, g_coop=g_coop, G_extern2excit=G_extern2excit) # 

    if model_opt == 'simple':
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, weight_profile_45 = wm_model_modified_simplified_EC_LV_principal.simulate_wm_EC_LV_Principal_Neurons_reduced_2(sim_time=sim_time_duration, poisson_firing_rate=poisson_firing_rate, sigma_weight_profile=sigma_weight_profile, Jpos_excit2excit=Jpos_excit2excit, t_stimulus_start=t_stimulus_start, t_stimulus_duration=t_stimulus_duration, stimulus_center_deg=stimulus_center_deg,stimulus_width_deg=stimulus_width_deg,N_excitatory=N_excitatory,N_inhibitory=N_inhibitory,weight_scaling_factor=weight_scaling_factor,stimulus_strength=stimulus_strength, G_inhib2inhib=G_inhib2inhib, G_inhib2excit=G_inhib2excit, G_excit2excit=G_excit2excit, G_excit2inhib=G_excit2inhib, G_extern2excit=G_extern2excit) # 

    # Temp: Reduced model using AMPA channels only
    #rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, weight_profile_45 = wm_model_modified_simplified_EC_LV_principal.simulate_wm_EC_LV_Principal_Neurons_reduced(sim_time=sim_time_duration, poisson_firing_rate=poisson_firing_rate, sigma_weight_profile=sigma_weight_profile, Jpos_excit2excit=Jpos_excit2excit, t_stimulus_start=t_stimulus_start, t_stimulus_duration=t_stimulus_duration, stimulus_center_deg=stimulus_center_deg,stimulus_width_deg=stimulus_width_deg,N_excitatory=N_excitatory,N_inhibitory=N_inhibitory,weight_scaling_factor=weight_scaling_factor,stimulus_strength=stimulus_strength, G_inhib2inhib=G_inhib2inhib, G_inhib2excit=G_inhib2excit, G_excit2excit=G_excit2excit, G_excit2inhib=G_excit2inhib, g_coop=g_coop) # , G_extern2excit=G_extern2excit
    
    # Temp: Use simulate_wm with x0 the working parameters
    # rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, weight_profile_45 = wm_model_modified_simplified_EC_LV_principal.simulate_wm(sim_time=sim_time_duration, poisson_firing_rate=poisson_firing_rate, sigma_weight_profile=sigma_weight_profile, t_stimulus_start=t_stimulus_start, t_stimulus_duration=t_stimulus_duration, stimulus_center_deg=stimulus_center_deg,stimulus_width_deg=stimulus_width_deg,N_excitatory=N_excitatory,N_inhibitory=N_inhibitory,weight_scaling_factor=weight_scaling_factor,stimulus_strength=stimulus_strength,G_inhib2inhib=G_inhib2inhib, G_inhib2excit=G_inhib2excit, G_excit2excit=G_excit2excit, G_excit2inhib=G_excit2inhib, g_coop=g_coop) # , G_extern2excit=G_extern2excit
    
    # Temp: Use simulate_wm_reduced Reduced model using some AMPA channels
    # rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, weight_profile_45 = wm_model_modified_simplified_EC_LV_principal.simulate_wm_reduced(sim_time=sim_time_duration, poisson_firing_rate=poisson_firing_rate, sigma_weight_profile=sigma_weight_profile, t_stimulus_start=t_stimulus_start, t_stimulus_duration=t_stimulus_duration, stimulus_center_deg=stimulus_center_deg,stimulus_width_deg=stimulus_width_deg,N_excitatory=N_excitatory,N_inhibitory=N_inhibitory,weight_scaling_factor=weight_scaling_factor,stimulus_strength=stimulus_strength,G_inhib2inhib=G_inhib2inhib, G_inhib2excit=G_inhib2excit, G_excit2excit=G_excit2excit, G_excit2inhib=G_excit2inhib, g_coop=g_coop) # , G_extern2excit=G_extern2excit
    
    # Temp: Use simulate_wm_reduced2 Reduced model using AMPA channels only
    # rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, weight_profile_45 = wm_model_modified_simplified_EC_LV_principal.simulate_wm_reduced2(sim_time=sim_time_duration, poisson_firing_rate=poisson_firing_rate, sigma_weight_profile=sigma_weight_profile, t_stimulus_start=t_stimulus_start, t_stimulus_duration=t_stimulus_duration, stimulus_center_deg=stimulus_center_deg,stimulus_width_deg=stimulus_width_deg,N_excitatory=N_excitatory,N_inhibitory=N_inhibitory,weight_scaling_factor=weight_scaling_factor,stimulus_strength=stimulus_strength,G_inhib2inhib=G_inhib2inhib, G_inhib2excit=G_inhib2excit, G_excit2excit=G_excit2excit, G_excit2inhib=G_excit2inhib, g_coop=g_coop) # , G_extern2excit=G_extern2excit

    fitness_error_1 = calc_objective_value(
        spike_monitor_excit, N_excitatory, 
        spike_monitor_inhib, N_inhibitory, 
        t_stimulus_start = t_stimulus_start, 
        t_stimulus_duration = t_stimulus_duration, 
        sim_time_duration = sim_time_duration,
        t_window_width = 20 * ms, 
        snapshot_interval = 10 * ms, 
        W_desired =  W_desired,            # Desired bump width
        W_desired_inhib = W_desired_inhib, # Desired width of inhibitory neurons activity (all active)
        H_desired = H_desired,             # Desired bump angle
        A_desired = A_desired              # Desired bump min amplitude
    )
    fitness_error_2 = calc_objective_value(
        spike_monitor_excit, N_excitatory, 
        spike_monitor_inhib, N_inhibitory, 
        t_stimulus_start = t_stimulus_start, 
        t_stimulus_duration = t_stimulus_duration, 
        sim_time_duration = sim_time_duration,
        t_window_width = 20 * ms, 
        snapshot_interval = 10 * ms, 
        W_desired =  W_desired,            # Desired bump width
        W_desired_inhib = W_desired_inhib, # Desired width of inhibitory neurons activity (all active)
        H_desired = H_desired,             # Desired bump angle
        A_desired = A_desired              # Desired bump min amplitude
    )
    fitness_error = (fitness_error_1 + fitness_error_2) / 2.0
    return fitness_error
    
def optimise(option, model='EC_LV', N_excitatory=512, N_inhibitory = 128, poisson_firing_rate = 1.4, G_extern2excit=24, stimulus_strength=2):
    # Set the global variables
    global model_opt
    global N_excitatory_opt
    global N_inhibitory_opt
    global poisson_firing_rate_opt
    global G_extern2excit_opt
    global stimulus_strength_opt

    model_opt        = model
    N_excitatory_opt = N_excitatory
    N_inhibitory_opt = N_inhibitory
    poisson_firing_rate_opt = poisson_firing_rate
    G_extern2excit_opt = G_extern2excit
    stimulus_strength_opt = stimulus_strength
    
    if model_opt == 'EC_LV':
        bnds_xmax = [1, 100, 100, 100, 100]
        bnds_xmax = [0.01, 10, 10, 10,  7.01]
        bnds_xmin = [0.,  0.,  0.,  0., 6.99]
        
        # define the space of hyperparameters to search
        search_space = [Real(0, 1),  # G_inhib2inhib
                        Real(0, 10), # G_inhib2excit
                        Real(0, 10), # G_excit2excit
                        Real(0, 10), # G_excit2inhib
                        Real(0, 10)] # g_coop

        # Initial values
        #x0 = [3, 2, 6, 12, 24] # in nS
        #x0 = [1., 1., 1., 1., 1.] # in nS
        #x0 = [10., 10., 10., 10., 10.] # in 
        #x0 = [26.22005783,  3.03904163, 35.09859011, 27.15835804, 15.27118935]

        #x0 = [4.29601014, 49.97432949, 16.36761731, 24.0936459,  12.2064772]

        # Starting from random values
        x0 = [rand()*1, rand()*100, rand()*100, rand()*100, rand()*10]

        # Some good options found starting from random values

        # Results for optimising for network size N_exc=512 N_inh=128
        #x0 = [0.5564806,   7.44954479, 41.42353866,  1.13106756, 92.22800553] # f = 4.6344 longer blocks of spiking
        #x0 = [0.9175976,   3.7267002,  99.52660805, 14.73296054,  6.47559851] # f = 3.0889 interminent short blocks of spiking
        #x0 = [ 0.9175976,   3.7267002,  99.52660805, 14.73296054,  7.64443366] # f = 2.9192
        #x0 = [0.87381165,  3.7267002,  99.52660805, 14.73296054,  7.64443366] # f = 2.8953
        #x0 = [0.86635697,  3.7267002,  66.12077086, 14.73296054,  7.64443366] # f = 2.8326 This is used for the data collection

        # I searched with higher boungs but the results is the following three produced worse bump regardless of improved fitness. 
        #x0 = [0.85316189,   3.7267002,  185.50369186,  14.73296054,   7.64443366] # f = 2.8317
        #x0 = [0.86635697,   3.7267002,  238.49482067,  14.73296054,   7.64443366] # f = 2.8751
        #x0 = [0.86635697,   3.7267002,  224.07178134,  14.73296054,   7.64443366] # f = 2.7975 

        # Results for optimising for network size N_exc=1024 N_inh=256
        # x0 = [0.35831593,  0.93167505, 16.5301927,   3.68324014,  7.64443366]

        # Results for optimising for network size N_exc=2048 N_inh=512
        # x0 = [0.21658924, 0.93167505, 16.5301927, 3.68324014, 7.64443366] # This is used for the data collection scalled to 2048

        x0_dict = {}

        # Results for optimising for network size N_exc=4096 N_inh=1024
        # 

        # Results for optimising for network size N_exc=8192 N_inh=2048
        # x0 = [0.16387249,  0.86503126, 41.13857958,  3.60112371, 52.21751574]
        # x0 = [0.1912275,   0.93167505, 56.93515254,  6.66132,    19.7632547]

        # Manually tuned for network size N_exc=2048 N_inh=512
        x0 = [0.0001, 0.005, 0.1, 1.0, 7.0]
        x0_dict[2048] = x0
        
        # Manually tuned for network size N_exc=1024 N_inh=256
        x0 = [0.0002, 0.010, 0.2, 2.0, 7.0]
        x0_dict[1024] = x0

        # Manually tuned for network size N_exc=512 N_inh=128
        x0 = [0.0004, 0.020, 0.4, 4.0, 7.0]
        x0_dict[512] = x0

        # Manually tuned for network size N_exc=256 N_inh=64
        x0 = [0.0002, 0.040, 0.8, 8.0, 7.0]
        x0 = [0.0077, 0.040, 0.8, 8.0, 7.0]
        x0_dict[256] = x0

        # Manually tuned for network size N_exc=128 N_inh=32
        x0 = [0.0077, 0.040, 0.8, 8.0, 7.0]
        x0_dict[128] = x0

        # Manually tuned for network size N_exc=4096 N_inh=1024
        x0 = [0.00005, 0.0025, 0.05, 0.669812245, 7.0]
        x0_dict[4096] = x0

        # Manually tuned for network size N_exc=8192 N_inh=2048
        x0 = [0.000025, 0.00125, 0.025, 7.0]
        x0_dict[8192] = x0

        x0 = x0_dict[N_excitatory]

    elif model_opt == 'simple':
        bnds_xmax = [1, 100, 100, 100 ]
        bnds_xmax = [10, 100, 100, 100 ]
        bnds_xmax = [1000, 1000, 1000, 1000 ]
        bnds_xmax = [10000, 10000, 10000, 10000 ]
        bnds_xmax = [1, 10, 10, 10]
        bnds_xmax = [0.01, 10, 10, 10]
        bnds_xmin = [0.,  0.,  0.,  0.]
        
        # define the space of hyperparameters to search
        search_space = [Real(0, 1),  # G_inhib2inhib
                        Real(0, 10), # G_inhib2excit
                        Real(0, 10), # G_excit2excit
                        Real(0, 10)] # G_excit2inhib

        # Starting from random values
        x0 = [rand()*1, rand()*100, rand()*100, rand()*100]
        x0 = [rand()*1, rand()*1, rand()*1, rand()*1]
        
        # Intermediate solutions 
        #x0 = [ 8.59945306, 27.9774266,  78.676898,   78.27788478] # f = 15.1143
        #x0 = [2.38696367e+00, 6.75928742e-02, 9.16553022e+01, 1.30270070e+01] # f = 12.4867
        #x0 = [2.96741064e+00, 6.75928742e-02, 9.69622946e+01, 1.30270070e+01] # f = 12.6522
        
        x0_dict = {}
        
        # Manually tuned for network size N_exc=2048 N_inh=512
        x0 = [0.0001, 0.005, 0.3, 1.0]
        x0 = [0.0004, 0.005, 0.3, 1.0]
        x0_dict[2048] = x0
        
        # Manually tuned for network size N_exc=1024 N_inh=256
        x0 = [0.0002, 0.010, 0.6, 2.0]
        x0 = [0.0008, 0.010, 0.6, 2.0]
        x0 = [0.000753299518, 0.01, 0.6, 2.0]
        x0_dict[1024] = x0

        # Manually tuned for network size N_exc=512 N_inh=128
        x0 = [0.0004, 0.020, 1.2, 4.0] # x4
        x0 = [0.0012111, 0.015, 0.9, 3.0] # x3
        x0 = [0.0001, 0.020, 1.2, 4.0] # x4 except first
        x0 = [0.0000861990726, 0.0357004628, 3.24396393, 4.88193990] # x1
        x0 = [0.000114210265,  0.0357004628, 3.24396393, 5.26641833] # x1
        x0 = [0.000355507334,  0.0357004628, 3.24396393, 5.26641833]
        x0 = [0.000125733024,  0.0357004628, 3.24396393, 5.26641833]
        x0_dict[512] = x0

        # Manually tuned for network size N_exc=256 N_inh=64
        x0 = [0.0008, 0.040, 2.4, 8.0]
        x0 = [0.0032, 0.04,  3.4, 8.0]
        #x0 = [0.00545939215, 0.04, 3.40815238, 8.32985526]
        x0 = [0.00210380550, 0.04, 3.4, 8.0]
        x0_dict[256] = x0

        # Manually tuned for network size N_exc=4096 N_inh=1024
        x0 = [0.0002, 0.0025, 0.15, 0.5]
        x0_dict[4096] = x0

        # Manually tuned for network size N_exc=8192 N_inh=2048
        x0 = [0.0001, 0.00125, 0.075, 0.25]
        x0_dict[8192] = x0

        x0 = x0_dict[N_excitatory]
        
    if option == 1:
        # Values must be positive
        bnds=list(zip(bnds_xmin, bnds_xmax))
        res = minimize(objective_function, x0, method='SLSQP', bounds=bnds)

    if option == 2:
        # Values must be positive
        bnds = list(zip(bnds_xmin, bnds_xmax))
        res = minimize(objective_function, x0, method='TNC', bounds=bnds)
    
    # Global optimizer
    if option == 3 or option == 'basin_hopping':
        def disp_progress_basin_hopping(x, f, accept):
            if accept:
                print('fitness:', f, 'minimum detected', 'x =', x)
            
        class MyBounds(object):
            def __init__(self, xmax=bnds_xmax, xmin=bnds_xmin):
                self.xmax = np.array(xmax)
                self.xmin = np.array(xmin)
            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                tmax = bool(np.all(x <= self.xmax))
                tmin = bool(np.all(x >= self.xmin))
                return tmax and tmin
        mybounds = MyBounds()
        res = basinhopping(objective_function, x0, accept_test=mybounds, disp=True, callback=disp_progress_basin_hopping)

    # Global optimizer
    if option == 4 or option == 'dual_annealing':
        def disp_progress_dual_annealing(x, f, context):
            print('fitness:', f, 'minimum detected, type:', context, 'x =', x)
            
        xmax=bnds_xmax
        xmin=bnds_xmin
        res = dual_annealing(objective_function, x0=x0, bounds=list(zip(xmin, xmax)), no_local_search=True, callback=disp_progress_dual_annealing, initial_temp=40000) # initial_temp=40000

        
    # Global optimizer
    if option == 5 or option == 'gaussian_optimization':
        res = gp_minimize(
                func=objective_function, # the function to minimize
                dimensions=search_space, # the bounds on each dimension of x
                #base_estimator=gpr,      # the Gaussian process estimator. The default is a Matern kernel
                acq_func='EI',           # the acquisition function: expected improvement. The default is 'gp_hedge'
                xi=0.01,                 # exploitation-exploration trade-off
                n_calls=100,              # the number of evaluations of f
                n_initial_points=10,      # number of random points evaluations before approximating it with base_estimator
                x0=[x0],                   # list of lists of initial input points or None
                #x0=X_init.tolist(),      # initial samples
                #y0=-Y_init.ravel(),      # Evaluation of initial input points.
                verbose=True)
        # summarizing finding:
        print('Best Accuracy: %.3f' % res.fun)
        print('Best Parameters: ', res.x)
        #plot_convergence(res)
    
    print(res)

    
if __name__ == '__main__':
    fire.Fire(optimise)
