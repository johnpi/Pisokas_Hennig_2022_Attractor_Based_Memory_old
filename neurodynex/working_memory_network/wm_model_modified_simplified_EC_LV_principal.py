"""

2 November 2020: Build on previous versions to use the EC Layer V principal cells dynamics. 

Simplified to use only one neurotransmitter receptor type.

Implementation of a working memory model.
Literature:
Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X. J. (2000). Synaptic mechanisms and
network dynamics underlying spatial working memory in a cortical network model.
Cerebral Cortex, 10(9), 910-923.

Some parts of this implementation are inspired by material from
*Stanford University, BIOE 332: Large-Scale Neural Modeling, Kwabena Boahen & Tatiana Engel, 2013*,
online available.

Note: Most parameters differ from the original publication.
"""

# This file is part of the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

# This free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License 2.0 as published by the
# Free Software Foundation. You should have received a copy of the
# GNU General Public License along with the repository. If not,
# see http://www.gnu.org/licenses/.

# Should you reuse and publish the code for your own purposes,
# please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

import brian2 as b2
from brian2 import NeuronGroup, Synapses, PoissonInput, network_operation
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from brian2.units import *
from random import sample
from collections import deque
from neurodynex.tools import plot_tools
import numpy
import matplotlib.pyplot as plt
import math
from scipy.special import erf
from numpy.fft import rfft, irfft
from scipy.stats import skewnorm
# Include for the additive gaussian white noise function
from utility_functions import add_gaussian_white_noise_by_magnitude

b2.defaultclock.dt = 0.05 * b2.ms

# Define a new mathematical operator for string arithmetic string expressions
# https://brian2.readthedocs.io/en/stable/user/equations.html
@check_units(x1 = 1, x2 = 1, result = 1)
def min(x1, x2):
    """ Returns the pairwise minima of two arrays or the minimum of two numbers """
    return numpy.minimum(x1, x2)
    


def simulate_wm_simple(
        N_excitatory=1024,          N_inhibitory=256,
        N_extern_poisson=1000,      poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
        sigma_weight_profile=20.,   
        stimulus_center_deg=180,    stimulus_width_deg=40,   stimulus_strength=0.07 * b2.namp,
        t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
        distractor_center_deg=90,   distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
        t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
        G_inhib2inhib=.35 * 1.024 * b2.nS,
        G_inhib2excit=.35 * 1.336 * b2.nS,
        G_excit2excit=.35 * 0.381 * b2.nS,
        G_excit2inhib=.35 * 1.2 * 0.292 * b2.nS,
        monitored_subset_size=1024, sim_time=800. * b2.ms,
        synaptic_noise_amount=0.0,
        J_inhib2inhib    = 0.3 / 1000, 
        J_inhib2excit    = 0.3 / 10000,
        Jpos_excit2excit = 0.3 / 1000,
        J_excit2inhib    = 0.3 / 1000,
        J_ext2inhib   = 1.0, 
        J_ext2excit   = 1.0, 
        tau_excit     = None, # Default is 20.0 * b2.ms (tau_excit needs to be higher than tau_inhib for bump maintenance)
        tau_inhib     = None  # Default is 10.0 * b2.ms
        ):
    
    # specify the excitatory pyramidal cells:
    if tau_excit is None: 
        Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
        R_leak_excit = 40 * 10**6 *b2.ohm # membrane resistence
        tau_excit = R_leak_excit * Cm_excit # membrane time constant Default is 20.0 * b2.ms
    else:
        Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
        R_leak_excit = tau_excit / Cm_excit # membrane resistence

    G_leak_excit = 20.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 2.0 * b2.ms  # absolute refractory period

    # specify the inhibitory interneurons:
    if tau_inhib is None:
        Cm_inhib = 0.2 * b2.nF
        R_leak_inhib = 50 * 10**6 *b2.ohm # membrane resistence
        tau_inhib = R_leak_inhib * Cm_inhib # membrane time constant Default is 10.0 * b2.ms
    else:
        Cm_inhib = 0.2 * b2.nF
        R_leak_inhib = tau_inhib / Cm_inhib # membrane resistence

    G_leak_inhib = 20.0 * b2.nS
    E_leak_inhib = -70.0 * b2.mV
    v_firing_threshold_inhib = -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 2.0 * b2.ms
    
    ext2inhib_stength = J_ext2inhib * 1.0*b2.mV
    ext2excit_stength = J_ext2excit * 1.0*b2.mV
    
    # Calculate the stimulus center
    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = t_distractor_start + t_distractor_duration
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx,
                                                            distr_center_idx + distr_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = \
        [(Jneg_excit2excit +
          (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
         for j in range(N_excitatory)]
    
    # Add noise to the synaptic weights
    presyn_weight_kernel = add_gaussian_white_noise_by_magnitude(presyn_weight_kernel, synaptic_noise_amount)
    
    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

    # define the inhibitory population
    inhib_lif_dynamics = """
        dv/dt = (1/tau) * (-(v-v_rest) + R_leak * I_stim ) : volt (unless refractory)
        tau : second
        v_rest : volt
        R_leak : ohm
        I_stim : amp
    """

    # specify the excitatory population:
    excit_lif_dynamics = """
        dv/dt = (1/tau) * (-(v-v_rest) + R_leak * I_stim ) : volt (unless refractory)
        tau : second
        v_rest : volt
        R_leak : ohm
        I_stim : amp
        x : 1
    """

    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_lif_dynamics,
        threshold="v>v_firing_threshold_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    inhib_pop.R_leak = R_leak_inhib
    inhib_pop.v_rest = E_leak_inhib
    inhib_pop.tau = tau_inhib
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, inhib_pop, 'w : 1', on_pre="v_post -= w*volt", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    syn_inhib2inhib.w = J_inhib2inhib
    
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="v",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=ext2inhib_stength)


    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit, method="rk2")
    excit_pop.R_leak = R_leak_excit
    excit_pop.v_rest = E_leak_excit
    excit_pop.tau = tau_excit
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="v",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=ext2excit_stength)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, excit_pop, 'w : 1', on_pre="v_post -= w*volt", delay=0.0 * b2.ms)
    syn_inhib2excit.connect(p=1.0)
    syn_inhib2excit.w = J_inhib2excit

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop, 'w : 1', on_pre="v_post += w*volt", delay=0.0 * b2.ms)
    syn_excit2inhib.connect(p=1.0)
    syn_excit2inhib.w = J_excit2inhib
    
    # set the connections: UNSTRUCTURED excitatory to excitatory
    syn_excit2excit = Synapses(excit_pop, excit_pop, 'w : 1', on_pre="v_post += w*volt", delay=0.0 * b2.ms)
    excit_pop.x = 'i' # Set an index to each neuron
    Gain = Jpos_excit2excit
    syn_excit2excit.connect(p=1.0)
    syn_excit2excit.w = 'Gain * exp( -(360.0 * min(abs(x_pre-x_post), N_excitatory - abs(x_pre-x_post)) / N_excitatory)**2 / (2 * sigma_weight_profile**2))'
    #syn_excit2excit.w = 'Gain * exp( -(360.0 * (abs(x_pre-x_post)) / N_excitatory)**2 / (2 * sigma_weight_profile**2))'

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = distractor_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        idx_monitored_neurons = \
            [int(math.ceil(k))
             for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)

    b2.run(sim_time)
    return \
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45





def simulate_wm(
    N_excitatory=1024, N_inhibitory=256,
    N_extern_poisson=1000, poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
    sigma_weight_profile=20., Jpos_excit2excit=1.6,
    stimulus_center_deg=180, stimulus_width_deg=40, stimulus_strength=0.07 * b2.namp,
    t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
    distractor_center_deg=90, distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
    t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
    G_inhib2inhib=.35 * 1.024 * b2.nS,
    G_inhib2excit=.35 * 1.336 * b2.nS,
    G_excit2excit=.35 * 0.381 * b2.nS,
    G_excit2inhib=.35 * 1.2 * 0.292 * b2.nS,
    G_extern2inhib = 2.38 * b2.nS, # Temp for testing
    G_extern2excit = 3.1 * b2.nS, # Temp for testing
    monitored_subset_size=1024, sim_time=800. * b2.ms,
    synaptic_noise_amount=0.0):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +\- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time
    Returns:
       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    # specify the excitatory pyramidal cells:
    Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 25.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 2.0 * b2.ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction

    #
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib = 0.2 * b2.nF
    G_leak_inhib = 20.0 * b2.nS
    E_leak_inhib = -70.0 * b2.mV
    v_firing_threshold_inhib = -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 1.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = .9 * 2.0 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = .65 * 100.0 * b2.ms  # orig: 100
    tau_NMDA_x = .94 * 2.0 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    #G_extern2inhib = 2.38 * b2.nS # Temp removed
    #G_extern2excit = 3.1 * b2.nS # Temp removed

    # projectsions from the inhibitory populations
    G_inhib2inhib *= weight_scaling_factor
    G_inhib2excit *= weight_scaling_factor

    # projections from the excitatory population
    G_excit2excit *= weight_scaling_factor
    G_excit2inhib *= weight_scaling_factor  # todo: verify this scaling

    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = t_distractor_start + t_distractor_duration
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx,
                                                            distr_center_idx + distr_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = \
        [(Jneg_excit2excit +
          (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
         for j in range(N_excitatory)]

    # Add noise to the synaptic weights
    presyn_weight_kernel = add_gaussian_white_noise_by_magnitude(presyn_weight_kernel, synaptic_noise_amount)

    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

    # define the inhibitory population
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - G_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_lif_dynamics,
        threshold="v>v_firing_threshold_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1.0",
                            refractory=t_abs_refract_excit, method="rk2")
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               model="s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    syn_excit2inhib.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop,
    #        model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)

    # set the STRUCTURED recurrent input. use a network_operation
    @network_operation()
    def update_nmda_sum():
        fft_s_NMDA = rfft(excit_pop.s_NMDA)
        fft_s_NMDA_total = numpy.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
        s_NMDA_tot = irfft(fft_s_NMDA_total)
        excit_pop.s_NMDA_total_ = s_NMDA_tot

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = distractor_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        idx_monitored_neurons = \
            [int(math.ceil(k))
             for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)

    b2.run(sim_time)
    return \
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45


def simulate_wm_skewed(
    N_excitatory=1024, N_inhibitory=256,
    N_extern_poisson=1000, poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
    sigma_weight_profile=20., Jpos_excit2excit=1.6,
    stimulus_center_deg=180, stimulus_width_deg=40, stimulus_strength=0.07 * b2.namp,
    t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
    distractor_center_deg=90, distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
    t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
    G_inhib2inhib=.35 * 1.024 * b2.nS,
    G_inhib2excit=.35 * 1.336 * b2.nS,
    G_excit2excit=.35 * 0.381 * b2.nS,
    G_excit2inhib=.35 * 1.2 * 0.292 * b2.nS,
    G_extern2inhib = 2.38 * b2.nS, # Temp for testing
    G_extern2excit = 3.1 * b2.nS, # Temp for testing
    monitored_subset_size=1024, sim_time=800. * b2.ms,
    synaptic_noise_amount=0.0,
    weights_skewness = 0.0 # Default 0 no skewness, results in normal distribution. Causes systematic shift of the activity bump. It is the value for the a parameter of the skewnorm.pdf fucntion
    ):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +\- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time
    Returns:
       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    # specify the excitatory pyramidal cells:
    Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 25.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 2.0 * b2.ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction

    #
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib = 0.2 * b2.nF
    G_leak_inhib = 20.0 * b2.nS
    E_leak_inhib = -70.0 * b2.mV
    v_firing_threshold_inhib = -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 1.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = .9 * 2.0 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = .65 * 100.0 * b2.ms  # orig: 100
    tau_NMDA_x = .94 * 2.0 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    #G_extern2inhib = 2.38 * b2.nS # Temp removed
    #G_extern2excit = 3.1 * b2.nS # Temp removed

    # projectsions from the inhibitory populations
    G_inhib2inhib *= weight_scaling_factor
    G_inhib2excit *= weight_scaling_factor

    # projections from the excitatory population
    G_excit2excit *= weight_scaling_factor
    G_excit2inhib *= weight_scaling_factor  # todo: verify this scaling

    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = t_distractor_start + t_distractor_duration
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx,
                                                            distr_center_idx + distr_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    # Replaced this 
    #presyn_weight_kernel = \
    #    [(Jneg_excit2excit +
    #      (Jpos_excit2excit - Jneg_excit2excit) *
    #      math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
    #     for j in range(N_excitatory)]
    # with this
    y02 = numpy.array([skewnorm.pdf((360. * (j - N_excitatory / 2) / N_excitatory), a=weights_skewness, loc=0, scale=sigma_weight_profile) for j in range(N_excitatory)])
    y02 = Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * y02/numpy.max(y02) # Normalise to 0-1
    fft_presyn_weight_kernel_02 = rfft(y02)
    presyn_weight_kernel = deque(y02)   # deque: a bouble ended queue data structure
    rot_dist_02 = int(round(len(presyn_weight_kernel) / 2)) # (len(weight_profile_45)==1024==N_excitatory)/8 = 128
    presyn_weight_kernel.rotate(rot_dist_02)                # rotate elements of the deque by 128 positions
    presyn_weight_kernel = list(presyn_weight_kernel)

    # Add noise to the synaptic weights
    presyn_weight_kernel = add_gaussian_white_noise_by_magnitude(presyn_weight_kernel, synaptic_noise_amount)

    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

    # define the inhibitory population
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - G_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_lif_dynamics,
        threshold="v>v_firing_threshold_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1.0",
                            refractory=t_abs_refract_excit, method="rk2")
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               model="s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    syn_excit2inhib.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop,
    #        model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)

    # set the STRUCTURED recurrent input. use a network_operation
    @network_operation()
    def update_nmda_sum():
        fft_s_NMDA = rfft(excit_pop.s_NMDA)
        fft_s_NMDA_total = numpy.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
        s_NMDA_tot = irfft(fft_s_NMDA_total)
        excit_pop.s_NMDA_total_ = s_NMDA_tot

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = distractor_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        idx_monitored_neurons = \
            [int(math.ceil(k))
             for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)

    b2.run(sim_time)
    return \
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45



# Model using EC Layer V principal neuron dynamics model
def simulate_wm_EC_LV_Principal_Neurons(
        N_excitatory=1024,          N_inhibitory=256,
        N_extern_poisson=1000,      poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
        sigma_weight_profile=20.,   Jpos_excit2excit=1.6,
        stimulus_center_deg=180,    stimulus_width_deg=40,   stimulus_strength=0.07 * b2.namp,
        t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
        distractor_center_deg=90,   distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
        t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
        G_inhib2inhib=.35 * 1.024 * b2.nS,
        G_inhib2excit=.35 * 1.336 * b2.nS, # was wi=6nS
        G_excit2excit=.35 * 0.381 * b2.nS, # was we=6nS
        G_excit2inhib=.35 * 1.2 * 0.292 * b2.nS,
        G_extern2excit=24 * b2.nS,
        G_extern2inhib = 2.38 * b2.nS,
        g_coop = 0.400 * b2.nS,
        monitored_subset_size=1024, sim_time=800. * b2.ms,
        synaptic_noise_amount=0.0):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +\- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time

    Returns:

       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    # specify the excitatory EC Layer V principal cells:
    area = 20000*b2.umetre**2        # Neuron surface area (0.02mm^2)
    Cm = (1*b2.ufarad*b2.cm**-2) * area # Bilipid layer surface capacitance (200.0pF)
    # Neuron parameters
    ## Reversal potentials
    Ee = 0*b2.mV
    Ei = -70*b2.mV
    El = -63*b2.mV
    EK = -15*b2.mV    # -75*mV    # K reversal potential
    ENa = 115*b2.mV   # 55*mV    # Na reversal potential
    Ecoop = 100*b2.mV # Cooperative cluster channel reversal potential

    ## Cannel conductances
    we     = 6    *b2.nS               # Excitatory synaptic weight increment
    wi     = 2    *b2.nS # 67   *nS    # Inhibitory synaptic weight increment
    gl     = (0.001*b2.msiemens*b2.cm**-2) * area
    g_na   = (100  *b2.msiemens*b2.cm**-2) * area
    g_kd   = (135  *b2.msiemens*b2.cm**-2) * area
    #g_coop = (2    *b2.usiemens*b2.cm**-2) * area # =400pS # Cooperative cluster channel
    
    ## Time constants
    taue     = 5 *b2.ms # Excitatory synaptic weight time constant
    taui     = 10*b2.ms # Inhibitory synaptic weight time constant
    tau_coop = 100*b2.second # 100s # Cooperative channels time constant. The higher the slower the spike rate
    tau_mAChR = 5 *b2.ms

    ## Sustained channel state variables
    O_max = 100 # Max number of open channels
    slope = 200
    factor_inc = 1e13 #* 1e4
    factor_dec = 5e13 #* 1e4
    
    ## Threshold potential
    Vth = -10*mV # 0*b2.mV     # -50*mV     # -20*mV # Threshold potential

    # specify the excitatory principal cells:
    Cm_excit = Cm # 0.2 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = gl # 20.0 * b2.nS  # leak conductance
    E_leak_excit = El # -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = 10 * b2.mV # -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 3.0 * b2.ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction

    #
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib = Cm # 0.2 * b2.nF
    G_leak_inhib = gl # 20.0 * b2.nS
    E_leak_inhib = El # -70.0 * b2.mV
    v_firing_threshold_inhib = 10 * b2.mV # -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 2.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = 2.0 * b2.ms # 1.8 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = 65.0 * b2.ms  # orig: 100
    tau_NMDA_x = 1.88 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    #G_extern2inhib = 2.38 * b2.nS # was wi=6nS
    #G_extern2excit = 6 * b2.nS # 3.1 * b2.nS # was we=6nS

    # projectsions from the inhibitory populations
    G_inhib2inhib *= weight_scaling_factor
    G_inhib2excit *= weight_scaling_factor

    # projections from the excitatory population
    G_excit2excit *= weight_scaling_factor
    G_excit2inhib *= weight_scaling_factor  # todo: verify this scaling

    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = t_distractor_start + t_distractor_duration
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx,
                                                            distr_center_idx + distr_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = \
        [(Jneg_excit2excit +
          (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
         for j in range(N_excitatory)]
    
    # Add noise to the synaptic weights
    presyn_weight_kernel = add_gaussian_white_noise_by_magnitude(presyn_weight_kernel, synaptic_noise_amount)
    
    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

    # define the inhibitory population
    inhib_HH_dynamics = b2.Equations("""
        I_stim : amp
        dv/dt = (
                 - G_leak_inhib * (v-E_leak_excit)
                 - G_excit2excit * s_mAChR * (v-Ee)
                 - G_inhib2inhib * s_GABA * (v-E_GABA)
                 - G_excit2inhib * s_AMPA * (v-E_AMPA)
                 - g_na*(m*m)*h*(v-ENa)
                 - g_kd*(n*n)*(v-EK)
                 + g_coop*O_coop*mV
                 + I_stim
                 )/Cm_inhib : volt

        # Synaptic conductances
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_mAChR/dt = -s_mAChR/tau_mAChR : 1

        dO_coop_unbounded/dt = (factor_inc*((G_excit2excit * s_mAChR)/siemens))/tau_coop - (factor_dec*((G_inhib2excit * s_GABA)/siemens))/tau_coop : 1
        #O_coop = O_coop_unbounded * int(O_coop_unbounded>0) : 1
        O_coop = O_coop_unbounded * int(O_coop_unbounded>0) * int(O_coop_unbounded<O_max) + O_max * int(O_coop_unbounded>=O_max) : 1

        # State variables for Na
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        alpha_m = 0.32*(mV**-1)*(13.1*mV-v)/(exp((13.1*mV-v)/(4*mV))-1)/ms : Hz
        beta_m =  0.28*(mV**-1)*(v-40.1*mV)/(exp((v-40.1*mV)/(5*mV))-1)/ms : Hz
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        alpha_h = 0.128*exp((17*mV-v)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v)/(5*mV)))/ms : Hz

        # State variables for K
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        alpha_n = 0.016*(mV**-1)*(35.1*mV-v)/(exp((35.1*mV-v)/(5*mV))-1)/ms : Hz
        beta_n = .25*exp((20*mV-v)/(40*mV))/ms : Hz
        """)
    
    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_HH_dynamics,
        threshold="v>v_firing_threshold_inhib",
        # reset="v=v_reset_inhib",
        refractory=t_abs_refract_inhib,
        #method="rk2"
        method="exponential_euler")
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    inhib_pop.v = numpy.random.uniform(E_leak_inhib / b2.mV - 5, high=E_leak_inhib / b2.mV + 5, size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the Hodgkin-Huxley excitatory population: EC Layer V principal neuron:
    excit_HH_dynamics_EC_LV = b2.Equations("""
        I_stim : amp
        dv/dt = (
                 - G_leak_excit * (v-E_leak_excit)
                 - G_excit2excit * s_mAChR * (v-Ee)
                 - G_inhib2excit * s_GABA * (v-E_GABA)
                 - G_extern2excit * s_AMPA * (v-E_AMPA)
                 - g_na*(m*m)*h*(v-ENa)
                 - g_kd*(n*n)*(v-EK)
                 + g_coop*O_coop*mV
                 + I_stim
                 )/Cm_excit : volt

        # Synaptic conductances
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_mAChR/dt = -s_mAChR/tau_mAChR : 1

        dO_coop_unbounded/dt = (factor_inc*((G_excit2excit * s_mAChR)/siemens))/tau_coop - (factor_dec*((G_inhib2excit * s_GABA)/siemens))/tau_coop : 1
        #O_coop = O_coop_unbounded * int(O_coop_unbounded>0) : 1
        O_coop = O_coop_unbounded * int(O_coop_unbounded>0) * int(O_coop_unbounded<O_max) + O_max * int(O_coop_unbounded>=O_max) : 1

        # State variables for Na
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        alpha_m = 0.32*(mV**-1)*(13.1*mV-v)/(exp((13.1*mV-v)/(4*mV))-1)/ms : Hz
        beta_m =  0.28*(mV**-1)*(v-40.1*mV)/(exp((v-40.1*mV)/(5*mV))-1)/ms : Hz
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        alpha_h = 0.128*exp((17*mV-v)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v)/(5*mV)))/ms : Hz

        # State variables for K
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        alpha_n = 0.016*(mV**-1)*(35.1*mV-v)/(exp((35.1*mV-v)/(5*mV))-1)/ms : Hz
        beta_n = .25*exp((20*mV-v)/(40*mV))/ms : Hz

        x : 1
        """)
    
    excit_pop = NeuronGroup(N_excitatory, model=excit_HH_dynamics_EC_LV,
                            threshold="v>v_firing_threshold_excit",
                            # reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit,
                            #method="rk2"
                            method="exponential_euler")
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV,
                                       high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.v = numpy.random.uniform(E_leak_excit / b2.mV - 5, high=E_leak_excit / b2.mV + 5, size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson,
                                   rate=poisson_firing_rate,
                                   weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               on_pre="s_AMPA += 1.0")
    syn_excit2inhib.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop,
    #        model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)

    # set the connections: UNSTRUCTURED excitatory to excitatory
    #syn_excit2excit = Synapses(excit_pop, excit_pop, on_pre="s_mAChR += 1.0")
    syn_excit2excit = Synapses(excit_pop, excit_pop, 'w : 1', on_pre="s_mAChR += w")
    #syn_excit2excit.connect(condition="i!=j", p=1.)
    syn_excit2excit.connect(p=1.)
    excit_pop.x = 'i' # Set an index to each neuron
    Gain = Jpos_excit2excit
    Gain = Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit)
    syn_excit2excit.w = 'Gain * exp( -(360.0 * min(abs(x_pre-x_post), N_excitatory - abs(x_pre-x_post)) / N_excitatory)**2 / (2 * sigma_weight_profile**2))'

    # # set the STRUCTURED recurrent input. use a network_operation
    # @network_operation()
    # def update_nmda_sum():
    #     fft_s_NMDA = rfft(excit_pop.s_NMDA)
    #     fft_s_NMDA_total = numpy.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
    #     s_NMDA_tot = irfft(fft_s_NMDA_total)
    #     excit_pop.s_NMDA_total_ = s_NMDA_tot

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = distractor_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        if nr_monitored < N:
            idx_monitored_neurons = [int(math.ceil(k)) for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        else:
            idx_monitored_neurons = range(0, N)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        #voltage_monitor = StateMonitor(pop, "True", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)

    b2.run(sim_time)
    return \
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45



# This version introduces the external noise to the inhibitory population which should have been there but somehow missed.
# Model using EC Layer V principal neuron dynamics model
def simulate_wm_EC_LV_Principal_Neurons_1(
        N_excitatory=1024,          N_inhibitory=256,
        N_extern_poisson=1000,      poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
        sigma_weight_profile=20.,   Jpos_excit2excit=1.6,
        stimulus_center_deg=180,    stimulus_width_deg=40,   stimulus_strength=0.07 * b2.namp,
        t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
        distractor_center_deg=90,   distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
        t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
        G_inhib2inhib=.35 * 1.024 * b2.nS,
        G_inhib2excit=.35 * 1.336 * b2.nS, # was wi=6nS
        G_excit2excit=.35 * 0.381 * b2.nS, # was we=6nS
        G_excit2inhib=.35 * 1.2 * 0.292 * b2.nS,
        G_extern2excit=24 * b2.nS,
        G_extern2inhib = 2.38 * b2.nS,
        g_coop = 0.400 * b2.nS,
        monitored_subset_size=1024, sim_time=800. * b2.ms,
        synaptic_noise_amount=0.0):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +\- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time

    Returns:

       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    # specify the excitatory EC Layer V principal cells:
    area = 20000*b2.umetre**2        # Neuron surface area (0.02mm^2)
    Cm = (1*b2.ufarad*b2.cm**-2) * area # Bilipid layer surface capacitance (200.0pF)
    # Neuron parameters
    ## Reversal potentials
    Ee = 0*b2.mV
    Ei = -70*b2.mV
    El = -63*b2.mV
    EK = -15*b2.mV    # -75*mV    # K reversal potential
    ENa = 115*b2.mV   # 55*mV    # Na reversal potential
    Ecoop = 100*b2.mV # Cooperative cluster channel reversal potential

    ## Cannel conductances
    we     = 6    *b2.nS               # Excitatory synaptic weight increment
    wi     = 2    *b2.nS # 67   *nS    # Inhibitory synaptic weight increment
    gl     = (0.001*b2.msiemens*b2.cm**-2) * area
    g_na   = (100  *b2.msiemens*b2.cm**-2) * area
    g_kd   = (135  *b2.msiemens*b2.cm**-2) * area
    #g_coop = (2    *b2.usiemens*b2.cm**-2) * area # =400pS # Cooperative cluster channel
    
    ## Time constants
    taue     = 5 *b2.ms # Excitatory synaptic weight time constant
    taui     = 10*b2.ms # Inhibitory synaptic weight time constant
    tau_coop = 100*b2.second # 100s # Cooperative channels time constant. The higher the slower the spike rate
    tau_mAChR = 5 *b2.ms

    ## Sustained channel state variables
    O_max = 100 # Max number of open channels
    slope = 200
    factor_inc = 1e13 #* 1e4
    factor_dec = 5e13 #* 1e4
    
    ## Threshold potential
    Vth = -10*mV # 0*b2.mV     # -50*mV     # -20*mV # Threshold potential

    # specify the excitatory principal cells:
    Cm_excit = Cm # 0.2 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = gl # 20.0 * b2.nS  # leak conductance
    E_leak_excit = El # -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = 10 * b2.mV # -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 3.0 * b2.ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction

    #
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib = Cm # 0.2 * b2.nF
    G_leak_inhib = gl # 20.0 * b2.nS
    E_leak_inhib = El # -70.0 * b2.mV
    v_firing_threshold_inhib = 10 * b2.mV # -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 2.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = 2.0 * b2.ms # 1.8 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = 65.0 * b2.ms  # orig: 100
    tau_NMDA_x = 1.88 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    #G_extern2inhib = 2.38 * b2.nS # was wi=6nS
    #G_extern2excit = 6 * b2.nS # 3.1 * b2.nS # was we=6nS

    # projectsions from the inhibitory populations
    G_inhib2inhib *= weight_scaling_factor
    G_inhib2excit *= weight_scaling_factor

    # projections from the excitatory population
    G_excit2excit *= weight_scaling_factor
    G_excit2inhib *= weight_scaling_factor  # todo: verify this scaling

    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = t_distractor_start + t_distractor_duration
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx,
                                                            distr_center_idx + distr_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = \
        [(Jneg_excit2excit +
          (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
         for j in range(N_excitatory)]
    
    # Add noise to the synaptic weights
    presyn_weight_kernel = add_gaussian_white_noise_by_magnitude(presyn_weight_kernel, synaptic_noise_amount)
    
    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

    # define the inhibitory population
    inhib_HH_dynamics = b2.Equations("""
        I_stim : amp
        dv/dt = (
                 - G_leak_inhib * (v-E_leak_excit)
                 - G_extern2inhib * s_AMPAext * (v-E_AMPA)
                 - G_excit2excit * s_mAChR * (v-Ee)
                 - G_inhib2inhib * s_GABA * (v-E_GABA)
                 - G_excit2inhib * s_AMPA * (v-E_AMPA)
                 - g_na*(m*m)*h*(v-ENa)
                 - g_kd*(n*n)*(v-EK)
                 + g_coop*O_coop*mV
                 + I_stim
                 )/Cm_inhib : volt

        # Synaptic conductances
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_AMPAext/dt = -s_AMPAext/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_mAChR/dt = -s_mAChR/tau_mAChR : 1

        dO_coop_unbounded/dt = (factor_inc*((G_excit2excit * s_mAChR)/siemens))/tau_coop - (factor_dec*((G_inhib2excit * s_GABA)/siemens))/tau_coop : 1
        #O_coop = O_coop_unbounded * int(O_coop_unbounded>0) : 1
        O_coop = O_coop_unbounded * int(O_coop_unbounded>0) * int(O_coop_unbounded<O_max) + O_max * int(O_coop_unbounded>=O_max) : 1

        # State variables for Na
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        alpha_m = 0.32*(mV**-1)*(13.1*mV-v)/(exp((13.1*mV-v)/(4*mV))-1)/ms : Hz
        beta_m =  0.28*(mV**-1)*(v-40.1*mV)/(exp((v-40.1*mV)/(5*mV))-1)/ms : Hz
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        alpha_h = 0.128*exp((17*mV-v)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v)/(5*mV)))/ms : Hz

        # State variables for K
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        alpha_n = 0.016*(mV**-1)*(35.1*mV-v)/(exp((35.1*mV-v)/(5*mV))-1)/ms : Hz
        beta_n = .25*exp((20*mV-v)/(40*mV))/ms : Hz
        """)
    
    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_HH_dynamics,
        threshold="v>v_firing_threshold_inhib",
        # reset="v=v_reset_inhib",
        refractory=t_abs_refract_inhib,
        #method="rk2"
        method="exponential_euler")
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    inhib_pop.v = numpy.random.uniform(E_leak_inhib / b2.mV - 5, high=E_leak_inhib / b2.mV + 5, size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPAext",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the Hodgkin-Huxley excitatory population: EC Layer V principal neuron:
    excit_HH_dynamics_EC_LV = b2.Equations("""
        I_stim : amp
        dv/dt = (
                 - G_leak_excit * (v-E_leak_excit)
                 - G_excit2excit * s_mAChR * (v-Ee)
                 - G_inhib2excit * s_GABA * (v-E_GABA)
                 - G_extern2excit * s_AMPA * (v-E_AMPA)
                 - g_na*(m*m)*h*(v-ENa)
                 - g_kd*(n*n)*(v-EK)
                 + g_coop*O_coop*mV
                 + I_stim
                 )/Cm_excit : volt

        # Synaptic conductances
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_mAChR/dt = -s_mAChR/tau_mAChR : 1

        dO_coop_unbounded/dt = (factor_inc*((G_excit2excit * s_mAChR)/siemens))/tau_coop - (factor_dec*((G_inhib2excit * s_GABA)/siemens))/tau_coop : 1
        #O_coop = O_coop_unbounded * int(O_coop_unbounded>0) : 1
        O_coop = O_coop_unbounded * int(O_coop_unbounded>0) * int(O_coop_unbounded<O_max) + O_max * int(O_coop_unbounded>=O_max) : 1

        # State variables for Na
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        alpha_m = 0.32*(mV**-1)*(13.1*mV-v)/(exp((13.1*mV-v)/(4*mV))-1)/ms : Hz
        beta_m =  0.28*(mV**-1)*(v-40.1*mV)/(exp((v-40.1*mV)/(5*mV))-1)/ms : Hz
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        alpha_h = 0.128*exp((17*mV-v)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v)/(5*mV)))/ms : Hz

        # State variables for K
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        alpha_n = 0.016*(mV**-1)*(35.1*mV-v)/(exp((35.1*mV-v)/(5*mV))-1)/ms : Hz
        beta_n = .25*exp((20*mV-v)/(40*mV))/ms : Hz

        x : 1
        """)
    
    excit_pop = NeuronGroup(N_excitatory, model=excit_HH_dynamics_EC_LV,
                            threshold="v>v_firing_threshold_excit",
                            # reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit,
                            #method="rk2"
                            method="exponential_euler")
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV,
                                       high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.v = numpy.random.uniform(E_leak_excit / b2.mV - 5, high=E_leak_excit / b2.mV + 5, size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson,
                                   rate=poisson_firing_rate,
                                   weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               on_pre="s_AMPA += 1.0")
    syn_excit2inhib.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop,
    #        model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)

    # set the connections: UNSTRUCTURED excitatory to excitatory
    #syn_excit2excit = Synapses(excit_pop, excit_pop, on_pre="s_mAChR += 1.0")
    syn_excit2excit = Synapses(excit_pop, excit_pop, 'w : 1', on_pre="s_mAChR += w")
    #syn_excit2excit.connect(condition="i!=j", p=1.)
    syn_excit2excit.connect(p=1.)
    excit_pop.x = 'i' # Set an index to each neuron
    Gain = Jpos_excit2excit
    Gain = Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit)
    syn_excit2excit.w = 'Gain * exp( -(360.0 * min(abs(x_pre-x_post), N_excitatory - abs(x_pre-x_post)) / N_excitatory)**2 / (2 * sigma_weight_profile**2))'

    # # set the STRUCTURED recurrent input. use a network_operation
    # @network_operation()
    # def update_nmda_sum():
    #     fft_s_NMDA = rfft(excit_pop.s_NMDA)
    #     fft_s_NMDA_total = numpy.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
    #     s_NMDA_tot = irfft(fft_s_NMDA_total)
    #     excit_pop.s_NMDA_total_ = s_NMDA_tot

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = distractor_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        if nr_monitored < N:
            idx_monitored_neurons = [int(math.ceil(k)) for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        else:
            idx_monitored_neurons = range(0, N)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        #voltage_monitor = StateMonitor(pop, "True", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)

    b2.run(sim_time)
    return \
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45





# This is not working it was just an attempt
# Model using EC Layer V principal neuron dynamics model
def simulate_wm_EC_LV_Principal_Neurons_2(
        N_excitatory=1024,          N_inhibitory=256,
        N_extern_poisson=1000,      poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
        sigma_weight_profile=20.,   Jpos_excit2excit=1.6,
        stimulus_center_deg=180,    stimulus_width_deg=40,   stimulus_strength=0.07 * b2.namp,
        t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
        distractor_center_deg=90,   distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
        t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
        G_inhib2inhib=.35 * 1.024 * b2.nS,
        G_inhib2excit=.35 * 1.336 * b2.nS, # was wi=6nS
        G_excit2excit=.35 * 0.381 * b2.nS, # was we=6nS
        G_excit2inhib=.35 * 1.2 * 0.292 * b2.nS,
        G_extern2inhib = 2.38 * b2.nS,
        #G_extern2excit=24 * b2.nS,
        G_extern2excit = 3.1 * b2.nS, # Temp for testing
        g_coop = 0.400 * b2.nS,
        monitored_subset_size=1024, sim_time=800. * b2.ms,
        synaptic_noise_amount=0.0):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +\- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time

    Returns:

       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    # specify the excitatory EC Layer V principal cells:
    area = 20000*b2.umetre**2        # Neuron surface area (0.02mm^2)
    Cm = (1*b2.ufarad*b2.cm**-2) * area # Bilipid layer surface capacitance (200.0pF)
    # Neuron parameters
    ## Reversal potentials
    Ee = 0*b2.mV
    Ei = -70*b2.mV
    El = -63*b2.mV
    EK = -15*b2.mV    # -75*mV    # K reversal potential
    ENa = 115*b2.mV   # 55*mV    # Na reversal potential
    Ecoop = 100*b2.mV # Cooperative cluster channel reversal potential

    ## Cannel conductances
    we     = 6    *b2.nS               # Excitatory synaptic weight increment
    wi     = 2    *b2.nS # 67   *nS    # Inhibitory synaptic weight increment
    gl     = (0.001*b2.msiemens*b2.cm**-2) * area
    g_na   = (100  *b2.msiemens*b2.cm**-2) * area
    g_kd   = (135  *b2.msiemens*b2.cm**-2) * area
    #g_coop = (2    *b2.usiemens*b2.cm**-2) * area # =400pS # Cooperative cluster channel
    
    ## Time constants
    taue     = 5 *b2.ms # Excitatory synaptic weight time constant
    taui     = 10*b2.ms # Inhibitory synaptic weight time constant
    tau_coop = 100*b2.second # 100s # Cooperative channels time constant. The higher the slower the spike rate
    tau_mAChR = 5 *b2.ms

    ## Sustained channel state variables
    O_max = 100 # Max number of open channels
    slope = 200
    factor_inc = 1e13 #* 1e4
    factor_dec = 5e13 #* 1e4
    
    ## Threshold potential
    Vth = -10*mV # 0*b2.mV     # -50*mV     # -20*mV # Threshold potential

    # specify the excitatory principal cells:
    Cm_excit = Cm # 0.2 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = gl # 20.0 * b2.nS  # leak conductance
    E_leak_excit = El # -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = 10 * b2.mV # -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 3.0 * b2.ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction

    #
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib = Cm # 0.2 * b2.nF
    G_leak_inhib = gl # 20.0 * b2.nS
    E_leak_inhib = El # -70.0 * b2.mV
    v_firing_threshold_inhib = 10 * b2.mV # -50.0 * b2.mV
    v_firing_threshold_inhib = -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 2.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = 2.0 * b2.ms # 1.8 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = 65.0 * b2.ms  # orig: 100
    tau_NMDA_x = 1.88 * b2.ms
    tau_NMDA_y = 1.88 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    #G_extern2inhib = 2.38 * b2.nS # was wi=6nS
    #G_extern2excit = 6 * b2.nS # 3.1 * b2.nS # was we=6nS

    # projectsions from the inhibitory populations
    G_inhib2inhib *= weight_scaling_factor
    G_inhib2excit *= weight_scaling_factor

    # projections from the excitatory population
    G_excit2excit *= weight_scaling_factor
    G_excit2inhib *= weight_scaling_factor  # todo: verify this scaling

    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = t_distractor_start + t_distractor_duration
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx,
                                                            distr_center_idx + distr_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = \
        [(Jneg_excit2excit +
          (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
         for j in range(N_excitatory)]
    
    # Add noise to the synaptic weights
    presyn_weight_kernel = add_gaussian_white_noise_by_magnitude(presyn_weight_kernel, synaptic_noise_amount)
    
    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

    # define the inhibitory population
    inhib_HH_dynamics = b2.Equations("""
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
                 - G_leak_inhib * (v-E_leak_excit)
                 - G_extern2inhib * s_AMPA * (v-E_AMPA)
                 - G_inhib2inhib * s_GABA * (v-E_GABA)
                 - G_excit2inhib * s_AMPA * (v-E_AMPA)
                 - G_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
                 - G_excit2excit * s_mAChR * (v-Ee)
                 - g_na*(m*m)*h*(v-ENa)
                 - g_kd*(n*n)*(v-EK)
                 + g_coop*O_coop*mV
                 + I_stim
                 )/Cm_inhib : volt

        # Synaptic conductances
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_mAChR/dt = -s_mAChR/tau_mAChR : 1

        dO_coop_unbounded/dt = (factor_inc*((G_excit2excit * s_mAChR)/siemens))/tau_coop - (factor_dec*((G_inhib2excit * s_GABA)/siemens))/tau_coop : 1
        #O_coop = O_coop_unbounded * int(O_coop_unbounded>0) : 1
        O_coop = O_coop_unbounded * int(O_coop_unbounded>0) * int(O_coop_unbounded<O_max) + O_max * int(O_coop_unbounded>=O_max) : 1

        # State variables for Na
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        alpha_m = 0.32*(mV**-1)*(13.1*mV-v)/(exp((13.1*mV-v)/(4*mV))-1)/ms : Hz
        beta_m =  0.28*(mV**-1)*(v-40.1*mV)/(exp((v-40.1*mV)/(5*mV))-1)/ms : Hz
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        alpha_h = 0.128*exp((17*mV-v)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v)/(5*mV)))/ms : Hz

        # State variables for K
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        alpha_n = 0.016*(mV**-1)*(35.1*mV-v)/(exp((35.1*mV-v)/(5*mV))-1)/ms : Hz
        beta_n = .25*exp((20*mV-v)/(40*mV))/ms : Hz
        """)
    
    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_HH_dynamics,
        threshold="v>v_firing_threshold_inhib",
        # reset="v=v_reset_inhib",
        refractory=t_abs_refract_inhib,
        #method="rk2"
        method="exponential_euler")
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    inhib_pop.v = numpy.random.uniform(E_leak_inhib / b2.mV - 5, high=E_leak_inhib / b2.mV + 5, size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the Hodgkin-Huxley excitatory population: EC Layer V principal neuron:
    excit_HH_dynamics_EC_LV = b2.Equations("""
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
                 - G_leak_excit * (v-E_leak_excit)
                 - G_extern2excit * s_AMPA * (v-E_AMPA)
                 - G_inhib2excit * s_GABA * (v-E_GABA)
                 - G_excit2excit * s_mAChR * (v-Ee)
                 - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
                 - g_na*(m*m)*h*(v-ENa)
                 - g_kd*(n*n)*(v-EK)
                 + g_coop*O_coop*mV
                 + I_stim
                 )/Cm_excit : volt

        # Synaptic conductances
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_mAChR/dt = -s_mAChR/tau_mAChR : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * y * (1-s_NMDA) : 1
        dy/dt = -y/tau_NMDA_y : 1

        dO_coop_unbounded/dt = (factor_inc*((G_excit2excit * s_mAChR)/siemens))/tau_coop - (factor_dec*((G_inhib2excit * s_GABA)/siemens))/tau_coop : 1
        #O_coop = O_coop_unbounded * int(O_coop_unbounded>0) : 1
        O_coop = O_coop_unbounded * int(O_coop_unbounded>0) * int(O_coop_unbounded<O_max) + O_max * int(O_coop_unbounded>=O_max) : 1

        # State variables for Na
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        alpha_m = 0.32*(mV**-1)*(13.1*mV-v)/(exp((13.1*mV-v)/(4*mV))-1)/ms : Hz
        beta_m =  0.28*(mV**-1)*(v-40.1*mV)/(exp((v-40.1*mV)/(5*mV))-1)/ms : Hz
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        alpha_h = 0.128*exp((17*mV-v)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v)/(5*mV)))/ms : Hz

        # State variables for K
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        alpha_n = 0.016*(mV**-1)*(35.1*mV-v)/(exp((35.1*mV-v)/(5*mV))-1)/ms : Hz
        beta_n = .25*exp((20*mV-v)/(40*mV))/ms : Hz

        x : 1
        """)
    
    excit_pop = NeuronGroup(N_excitatory, model=excit_HH_dynamics_EC_LV,
                            threshold="v>v_firing_threshold_excit",
                            # reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit,
                            #method="rk2"
                            method="exponential_euler")
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV,
                                       high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.v = numpy.random.uniform(E_leak_excit / b2.mV - 5, high=E_leak_excit / b2.mV + 5, size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson,
                                   rate=poisson_firing_rate,
                                   weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    #syn_excit2inhib = Synapses(excit_pop, inhib_pop,
    #                           on_pre="s_AMPA += 1.0")
    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               model="s_NMDA_total_post = s_NMDA_pre : 1 (summed)")
    syn_excit2inhib.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop,
    #        model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)

    # set the connections: UNSTRUCTURED excitatory to excitatory
    #syn_excit2excit = Synapses(excit_pop, excit_pop, on_pre="s_mAChR += 1.0")
    syn_excit2excit = Synapses(excit_pop, excit_pop, 'w : 1', on_pre="s_mAChR += w")
    #syn_excit2excit.connect(condition="i!=j", p=1.)
    syn_excit2excit.connect(p=1.)
    excit_pop.x = 'i' # Set an index to each neuron
    Gain = Jpos_excit2excit
    Gain = Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit)
    syn_excit2excit.w = 'Gain * exp( -(360.0 * min(abs(x_pre-x_post), N_excitatory - abs(x_pre-x_post)) / N_excitatory)**2 / (2 * sigma_weight_profile**2))'

    # set the STRUCTURED recurrent input. use a network_operation
    @network_operation()
    def update_nmda_sum():
        fft_s_NMDA = rfft(excit_pop.s_NMDA)
        fft_s_NMDA_total = numpy.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
        s_NMDA_tot = irfft(fft_s_NMDA_total)
        excit_pop.s_NMDA_total_ = s_NMDA_tot

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = distractor_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        if nr_monitored < N:
            idx_monitored_neurons = [int(math.ceil(k)) for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        else:
            idx_monitored_neurons = range(0, N)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        #voltage_monitor = StateMonitor(pop, "True", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)

    b2.run(sim_time)
    return \
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45








# Model using EC Layer V principal neuron dynamics model
def simulate_wm_EC_LV_Principal_Neurons_reduced_2(
        N_excitatory=1024,          N_inhibitory=256,
        N_extern_poisson=1000,      poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
        sigma_weight_profile=20.,   Jpos_excit2excit=1.6,
        stimulus_center_deg=180,    stimulus_width_deg=40,   stimulus_strength=0.07 * b2.namp,
        t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
        distractor_center_deg=90,   distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
        t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
        G_inhib2inhib=.35 * 1.024 * b2.nS,
        G_inhib2excit=.35 * 1.336 * b2.nS, # was wi=6nS
        G_excit2excit=.35 * 0.381 * b2.nS, # was we=6nS
        G_excit2inhib=.35 * 1.2 * 0.292 * b2.nS,
        G_extern2excit=24 * b2.nS,
        G_extern2inhib = 2.38 * b2.nS,
        g_coop = 0.400 * b2.nS,
        monitored_subset_size=1024, sim_time=800. * b2.ms,
        synaptic_noise_amount=0.0):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +\- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time

    Returns:

       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    # specify the excitatory EC Layer V principal cells:
    area = 20000*b2.umetre**2        # Neuron surface area (0.02mm^2)
    Cm = (1*b2.ufarad*b2.cm**-2) * area # Bilipid layer surface capacitance (200.0pF)
    # Neuron parameters
    ## Reversal potentials
    Ee = 0*b2.mV
    Ei = -70*b2.mV
    El = -63*b2.mV
    EK = -15*b2.mV    # -75*mV    # K reversal potential
    ENa = 115*b2.mV   # 55*mV    # Na reversal potential
    Ecoop = 100*b2.mV # Cooperative cluster channel reversal potential

    ## Cannel conductances
    we     = 6    *b2.nS               # Excitatory synaptic weight increment
    wi     = 2    *b2.nS # 67   *nS    # Inhibitory synaptic weight increment
    gl     = (0.001*b2.msiemens*b2.cm**-2) * area
    g_na   = (100  *b2.msiemens*b2.cm**-2) * area
    g_kd   = (135  *b2.msiemens*b2.cm**-2) * area
    #g_coop = (2    *b2.usiemens*b2.cm**-2) * area # =400pS # Cooperative cluster channel
    
    ## Time constants
    taue     = 5 *b2.ms # Excitatory synaptic weight time constant
    taui     = 10*b2.ms # Inhibitory synaptic weight time constant
    tau_coop = 100*b2.second # 100s # Cooperative channels time constant. The higher the slower the spike rate
    tau_mAChR = 5 *b2.ms

    ## Sustained channel state variables
    O_max = 100 # Max number of open channels
    slope = 200
    factor_inc = 1e13 #* 1e4
    factor_dec = 5e13 #* 1e4
    
    ## Threshold potential
    Vth = -10*mV # 0*b2.mV     # -50*mV     # -20*mV # Threshold potential

    # specify the excitatory principal cells:
    Cm_excit = Cm # 0.2 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = gl # 20.0 * b2.nS  # leak conductance
    E_leak_excit = El # -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = 10 * b2.mV # -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 3.0 * b2.ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction

    #
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib = Cm # 0.2 * b2.nF
    G_leak_inhib = gl # 20.0 * b2.nS
    E_leak_inhib = El # -70.0 * b2.mV
    v_firing_threshold_inhib = 10 * b2.mV # -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 2.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = 2.0 * b2.ms # 1.8 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = 65.0 * b2.ms  # orig: 100
    tau_NMDA_x = 1.88 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    #G_extern2inhib = 2.38 * b2.nS # was wi=6nS
    #G_extern2excit = 6 * b2.nS # 3.1 * b2.nS # was we=6nS

    # projectsions from the inhibitory populations
    G_inhib2inhib *= weight_scaling_factor
    G_inhib2excit *= weight_scaling_factor

    # projections from the excitatory population
    G_excit2excit *= weight_scaling_factor
    G_excit2inhib *= weight_scaling_factor  # todo: verify this scaling

    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = t_distractor_start + t_distractor_duration
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx,
                                                            distr_center_idx + distr_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = \
        [(Jneg_excit2excit +
          (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
         for j in range(N_excitatory)]
    
    # Add noise to the synaptic weights
    presyn_weight_kernel = add_gaussian_white_noise_by_magnitude(presyn_weight_kernel, synaptic_noise_amount)
    
    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

    # define the inhibitory population
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - G_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """
    # redefine the inhibitory population
    inhib_lif_dynamics = """
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """
    # redefine the inhibitory population
    inhib_HH_dynamics = b2.Equations("""
        I_stim : amp
        dv/dt = (
                 - G_leak_inhib * (v-E_leak_excit)
                 - G_excit2excit * s_mAChR * (v-Ee)
                 - G_inhib2inhib * s_GABA * (v-E_GABA)
                 - G_excit2inhib * s_AMPA * (v-E_AMPA)
                 - g_na*(m*m)*h*(v-ENa)
                 - g_kd*(n*n)*(v-EK)
                 #+ g_coop*O_coop*mV
                 + I_stim
                 )/Cm_inhib : volt

        # Synaptic conductances
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_mAChR/dt = -s_mAChR/tau_mAChR : 1

        dO_coop_unbounded/dt = (factor_inc*((G_excit2excit * s_mAChR)/siemens))/tau_coop - (factor_dec*((G_inhib2excit * s_GABA)/siemens))/tau_coop : 1
        #O_coop = O_coop_unbounded * int(O_coop_unbounded>0) : 1
        O_coop = O_coop_unbounded * int(O_coop_unbounded>0) * int(O_coop_unbounded<O_max) + O_max * int(O_coop_unbounded>=O_max) : 1

        # State variables for Na
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        alpha_m = 0.32*(mV**-1)*(13.1*mV-v)/(exp((13.1*mV-v)/(4*mV))-1)/ms : Hz
        beta_m =  0.28*(mV**-1)*(v-40.1*mV)/(exp((v-40.1*mV)/(5*mV))-1)/ms : Hz
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        alpha_h = 0.128*exp((17*mV-v)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v)/(5*mV)))/ms : Hz

        # State variables for K
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        alpha_n = 0.016*(mV**-1)*(35.1*mV-v)/(exp((35.1*mV-v)/(5*mV))-1)/ms : Hz
        beta_n = .25*exp((20*mV-v)/(40*mV))/ms : Hz
        """)
    
    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_HH_dynamics,
        threshold="v>v_firing_threshold_inhib",
        # reset="v=v_reset_inhib",
        refractory=t_abs_refract_inhib,
        #method="rk2"
        method="exponential_euler")
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    inhib_pop.v = numpy.random.uniform(E_leak_inhib / b2.mV - 5, high=E_leak_inhib / b2.mV + 5, size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """
    # respecify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """
    # respecify the Hodgkin-Huxley excitatory population: EC Layer V principal neuron:
    excit_HH_dynamics_EC_LV = b2.Equations("""
        I_stim : amp
        dv/dt = (
                 - G_leak_excit * (v-E_leak_excit)
                 - G_excit2excit * s_mAChR * (v-Ee)
                 - G_inhib2excit * s_GABA * (v-E_GABA)
                 - G_extern2excit * s_AMPA * (v-E_AMPA)
                 - g_na*(m*m)*h*(v-ENa)
                 - g_kd*(n*n)*(v-EK)
                 #+ g_coop*O_coop*mV
                 + I_stim
                 )/Cm_excit : volt

        # Synaptic conductances
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_mAChR/dt = -s_mAChR/tau_mAChR : 1

        dO_coop_unbounded/dt = (factor_inc*((G_excit2excit * s_mAChR)/siemens))/tau_coop - (factor_dec*((G_inhib2excit * s_GABA)/siemens))/tau_coop : 1
        #O_coop = O_coop_unbounded * int(O_coop_unbounded>0) : 1
        O_coop = O_coop_unbounded * int(O_coop_unbounded>0) * int(O_coop_unbounded<O_max) + O_max * int(O_coop_unbounded>=O_max) : 1

        # State variables for Na
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        alpha_m = 0.32*(mV**-1)*(13.1*mV-v)/(exp((13.1*mV-v)/(4*mV))-1)/ms : Hz
        beta_m =  0.28*(mV**-1)*(v-40.1*mV)/(exp((v-40.1*mV)/(5*mV))-1)/ms : Hz
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        alpha_h = 0.128*exp((17*mV-v)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v)/(5*mV)))/ms : Hz

        # State variables for K
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        alpha_n = 0.016*(mV**-1)*(35.1*mV-v)/(exp((35.1*mV-v)/(5*mV))-1)/ms : Hz
        beta_n = .25*exp((20*mV-v)/(40*mV))/ms : Hz

        x : 1
        """)
    
    excit_pop = NeuronGroup(N_excitatory, model=excit_HH_dynamics_EC_LV,
                            threshold="v>v_firing_threshold_excit",
                            # reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit,
                            #method="rk2"
                            method="exponential_euler")
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV,
                                       high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.v = numpy.random.uniform(E_leak_excit / b2.mV - 5, high=E_leak_excit / b2.mV + 5, size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson,
                                   rate=poisson_firing_rate,
                                   weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               on_pre="s_AMPA += 1.0")
    syn_excit2inhib.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop,
    #        model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)

    # set the connections: UNSTRUCTURED excitatory to excitatory
    #syn_excit2excit = Synapses(excit_pop, excit_pop, on_pre="s_mAChR += 1.0")
    syn_excit2excit = Synapses(excit_pop, excit_pop, 'w : 1', on_pre="s_mAChR += w")
    #syn_excit2excit.connect(condition="i!=j", p=1.)
    syn_excit2excit.connect(p=1.)
    excit_pop.x = 'i' # Set an index to each neuron
    Gain = Jpos_excit2excit
    Gain = Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit)
    syn_excit2excit.w = 'Gain * exp( -(360.0 * min(abs(x_pre-x_post), N_excitatory - abs(x_pre-x_post)) / N_excitatory)**2 / (2 * sigma_weight_profile**2))'

    # # set the STRUCTURED recurrent input. use a network_operation
    # @network_operation()
    # def update_nmda_sum():
    #     fft_s_NMDA = rfft(excit_pop.s_NMDA)
    #     fft_s_NMDA_total = numpy.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
    #     s_NMDA_tot = irfft(fft_s_NMDA_total)
    #     excit_pop.s_NMDA_total_ = s_NMDA_tot

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = distractor_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        if nr_monitored < N:
            idx_monitored_neurons = [int(math.ceil(k)) for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        else:
            idx_monitored_neurons = range(0, N)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        #voltage_monitor = StateMonitor(pop, "True", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)

    b2.run(sim_time)
    return \
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45






# Model using EC Layer V principal neuron dynamics model reduced
def simulate_wm_EC_LV_Principal_Neurons_reduced(
        N_excitatory=1024,          N_inhibitory=256,
        N_extern_poisson=1000,      poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
        sigma_weight_profile=20.,   Jpos_excit2excit=1.6,
        stimulus_center_deg=180,    stimulus_width_deg=40,   stimulus_strength=0.07 * b2.namp,
        t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
        distractor_center_deg=90,   distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
        t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
        G_inhib2inhib=.35 * 1.024 * b2.nS,
        G_inhib2excit=.35 * 1.336 * b2.nS, # was wi=6nS
        G_excit2excit=.35 * 0.381 * b2.nS, # was we=6nS
        G_excit2inhib=.35 * 1.2 * 0.292 * b2.nS,
        G_extern2excit=24 * b2.nS,
        G_extern2inhib = 2.38 * b2.nS,
        monitored_subset_size=1024, sim_time=800. * b2.ms,
        synaptic_noise_amount=0.0):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +\- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time

    Returns:

       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    # specify the excitatory EC Layer V principal cells:
    area = 20000*b2.umetre**2        # Neuron surface area (0.02mm^2)
    Cm = (1*b2.ufarad*b2.cm**-2) * area # Bilipid layer surface capacitance (200.0pF)
    # Neuron parameters
    ## Reversal potentials
    Ee = 0*b2.mV
    Ei = -70*b2.mV
    El = -63*b2.mV
    EK = -15*b2.mV    # -75*mV    # K reversal potential
    ENa = 115*b2.mV   # 55*mV    # Na reversal potential
    Ecoop = 100*b2.mV # Cooperative cluster channel reversal potential

    ## Cannel conductances
    we     = 6    *b2.nS               # Excitatory synaptic weight increment
    wi     = 2    *b2.nS # 67   *nS    # Inhibitory synaptic weight increment
    gl     = (0.001*b2.msiemens*b2.cm**-2) * area
    g_na   = (100  *b2.msiemens*b2.cm**-2) * area
    g_kd   = (135  *b2.msiemens*b2.cm**-2) * area
    g_coop = (2    *b2.usiemens*b2.cm**-2) * area # =400pS # Cooperative cluster channel
    
    ## Time constants
    taue     = 5 *b2.ms # Excitatory synaptic weight time constant
    taui     = 10*b2.ms # Inhibitory synaptic weight time constant
    tau_coop = 100*b2.second # 100s # Cooperative channels time constant. The higher the slower the spike rate
    tau_mAChR = 5 *b2.ms

    ## Sustained channel state variables
    O_max = 100 # Max number of open channels
    slope = 200
    factor_inc = 1e13 #* 1e4
    factor_dec = 5e13 #* 1e4
    
    ## Threshold potential
    Vth = -10*mV # 0*b2.mV     # -50*mV     # -20*mV # Threshold potential

    # specify the excitatory principal cells:
    Cm_excit = Cm # 0.2 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = gl # 20.0 * b2.nS  # leak conductance
    E_leak_excit = El # -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = 10 * b2.mV # -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 3.0 * b2.ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction

    #
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib = Cm # 0.2 * b2.nF
    G_leak_inhib = gl # 20.0 * b2.nS
    E_leak_inhib = El # -70.0 * b2.mV
    v_firing_threshold_inhib = 10 * b2.mV # -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 2.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = 2.0 * b2.ms # 1.8 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = 65.0 * b2.ms  # orig: 100
    tau_NMDA_x = 1.88 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    #G_extern2inhib = 2.38 * b2.nS # was wi=6nS
    #G_extern2excit = 6 * b2.nS # 3.1 * b2.nS # was we=6nS

    # projectsions from the inhibitory populations
    G_inhib2inhib *= weight_scaling_factor
    G_inhib2excit *= weight_scaling_factor

    # projections from the excitatory population
    G_excit2excit *= weight_scaling_factor
    G_excit2inhib *= weight_scaling_factor  # todo: verify this scaling

    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = t_distractor_start + t_distractor_duration
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx,
                                                            distr_center_idx + distr_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = \
        [(Jneg_excit2excit +
          (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
         for j in range(N_excitatory)]
    
    # Add noise to the synaptic weights
    presyn_weight_kernel = add_gaussian_white_noise_by_magnitude(presyn_weight_kernel, synaptic_noise_amount)
    
    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

    # define the inhibitory population
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - G_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """
    # redefine the inhibitory population
    inhib_lif_dynamics = """
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """
    # redefine the inhibitory population
    inhib_HH_dynamics = b2.Equations("""
        I_stim : amp
        dv/dt = (
                 - G_leak_inhib * (v-E_leak_excit)
                 - G_excit2excit * s_mAChR * (v-Ee)
                 - G_inhib2inhib * s_GABA * (v-E_GABA)
                 - G_excit2inhib * s_AMPA * (v-E_AMPA)
                 - g_na*(m*m)*h*(v-ENa)
                 - g_kd*(n*n)*(v-EK)
                 + g_coop*O_coop*mV
                 + I_stim
                 )/Cm_inhib : volt

        # Synaptic conductances
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_mAChR/dt = -s_mAChR/tau_mAChR : 1

        dO_coop_unbounded/dt = (factor_inc*((G_excit2excit * s_mAChR)/siemens))/tau_coop - (factor_dec*((G_inhib2excit * s_GABA)/siemens))/tau_coop : 1
        #O_coop = O_coop_unbounded * int(O_coop_unbounded>0) : 1
        O_coop = O_coop_unbounded * int(O_coop_unbounded>0) * int(O_coop_unbounded<O_max) + O_max * int(O_coop_unbounded>=O_max) : 1

        # State variables for Na
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        alpha_m = 0.32*(mV**-1)*(13.1*mV-v)/(exp((13.1*mV-v)/(4*mV))-1)/ms : Hz
        beta_m =  0.28*(mV**-1)*(v-40.1*mV)/(exp((v-40.1*mV)/(5*mV))-1)/ms : Hz
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        alpha_h = 0.128*exp((17*mV-v)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v)/(5*mV)))/ms : Hz

        # State variables for K
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        alpha_n = 0.016*(mV**-1)*(35.1*mV-v)/(exp((35.1*mV-v)/(5*mV))-1)/ms : Hz
        beta_n = .25*exp((20*mV-v)/(40*mV))/ms : Hz
        """)
    
    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_HH_dynamics,
        threshold="v>v_firing_threshold_inhib",
        # reset="v=v_reset_inhib",
        refractory=t_abs_refract_inhib,
        #method="rk2"
        method="exponential_euler")
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """
    # respecify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """
    # respecify the Hodgkin-Huxley excitatory population: EC Layer V principal neuron:
    excit_HH_dynamics_EC_LV = b2.Equations("""
        I_stim : amp
        dv/dt = (
                 - G_leak_excit * (v-E_leak_excit)
                 - G_excit2excit * s_mAChR * (v-Ee)
                 - G_inhib2excit * s_GABA * (v-E_GABA)
                 - G_extern2excit * s_AMPA * (v-E_AMPA)
                 - g_na*(m*m)*h*(v-ENa)
                 - g_kd*(n*n)*(v-EK)
                 + g_coop*O_coop*mV
                 + I_stim
                 )/Cm_excit : volt

        # Synaptic conductances
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_mAChR/dt = -s_mAChR/tau_mAChR : 1

        dO_coop_unbounded/dt = (factor_inc*((G_excit2excit * s_mAChR)/siemens))/tau_coop - (factor_dec*((G_inhib2excit * s_GABA)/siemens))/tau_coop : 1
        #O_coop = O_coop_unbounded * int(O_coop_unbounded>0) : 1
        O_coop = O_coop_unbounded * int(O_coop_unbounded>0) * int(O_coop_unbounded<O_max) + O_max * int(O_coop_unbounded>=O_max) : 1

        # State variables for Na
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        alpha_m = 0.32*(mV**-1)*(13.1*mV-v)/(exp((13.1*mV-v)/(4*mV))-1)/ms : Hz
        beta_m =  0.28*(mV**-1)*(v-40.1*mV)/(exp((v-40.1*mV)/(5*mV))-1)/ms : Hz
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        alpha_h = 0.128*exp((17*mV-v)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v)/(5*mV)))/ms : Hz

        # State variables for K
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        alpha_n = 0.016*(mV**-1)*(35.1*mV-v)/(exp((35.1*mV-v)/(5*mV))-1)/ms : Hz
        beta_n = .25*exp((20*mV-v)/(40*mV))/ms : Hz

        x : 1
        """)
    
    excit_pop = NeuronGroup(N_excitatory, model=excit_HH_dynamics_EC_LV,
                            threshold="v>v_firing_threshold_excit",
                            # reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit,
                            #method="rk2"
                            method="exponential_euler")
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV,
                                       high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson,
                                   rate=poisson_firing_rate,
                                   weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               on_pre="s_AMPA += 1.0")
    syn_excit2inhib.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop,
    #        model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)

    # set the connections: UNSTRUCTURED excitatory to excitatory
    #syn_excit2excit = Synapses(excit_pop, excit_pop, on_pre="s_mAChR += 1.0")
    syn_excit2excit = Synapses(excit_pop, excit_pop, 'w : 1', on_pre="s_AMPA += w")
    #syn_excit2excit.connect(condition="i!=j", p=1.)
    syn_excit2excit.connect(p=1.)
    excit_pop.x = 'i' # Set an index to each neuron
    Gain = Jpos_excit2excit
    Gain = Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit)
    syn_excit2excit.w = 'Gain * exp( -(360.0 * min(abs(x_pre-x_post), N_excitatory - abs(x_pre-x_post)) / N_excitatory)**2 / (2 * sigma_weight_profile**2))'

    # # set the STRUCTURED recurrent input. use a network_operation
    # @network_operation()
    # def update_nmda_sum():
    #     fft_s_NMDA = rfft(excit_pop.s_NMDA)
    #     fft_s_NMDA_total = numpy.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
    #     s_NMDA_tot = irfft(fft_s_NMDA_total)
    #     excit_pop.s_NMDA_total_ = s_NMDA_tot

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = distractor_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        idx_monitored_neurons = \
            [int(math.ceil(k))
             for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        #voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, True, record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)

    b2.run(sim_time)
    return \
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45




def simulate_wm_reduced(
    N_excitatory=1024, N_inhibitory=256,
    N_extern_poisson=1000, poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
    sigma_weight_profile=20., Jpos_excit2excit=1.6,
    stimulus_center_deg=180, stimulus_width_deg=40, stimulus_strength=0.07 * b2.namp,
    t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
    distractor_center_deg=90, distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
    t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
    G_inhib2inhib=.35 * 1.024 * b2.nS,
    G_inhib2excit=.35 * 1.336 * b2.nS,
    G_excit2excit=.35 * 0.381 * b2.nS,
    G_excit2inhib=.35 * 1.2 * 0.292 * b2.nS,
    G_extern2inhib = 2.38 * b2.nS, # Temp for testing
    G_extern2excit = 3.1 * b2.nS, # Temp for testing
    monitored_subset_size=1024, sim_time=800. * b2.ms,
    synaptic_noise_amount=0.0):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +\- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time
    Returns:
       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    # specify the excitatory pyramidal cells:
    Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 25.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 2.0 * b2.ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction

    #
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib = 0.2 * b2.nF
    G_leak_inhib = 20.0 * b2.nS
    E_leak_inhib = -70.0 * b2.mV
    v_firing_threshold_inhib = -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 1.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = .9 * 2.0 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = .65 * 100.0 * b2.ms  # orig: 100
    tau_NMDA_x = .94 * 2.0 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    #G_extern2inhib = 2.38 * b2.nS # Temp removed
    #G_extern2excit = 3.1 * b2.nS # Temp removed

    # projectsions from the inhibitory populations
    G_inhib2inhib *= weight_scaling_factor
    G_inhib2excit *= weight_scaling_factor

    # projections from the excitatory population
    G_excit2excit *= weight_scaling_factor
    G_excit2inhib *= weight_scaling_factor  # todo: verify this scaling

    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = t_distractor_start + t_distractor_duration
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx,
                                                            distr_center_idx + distr_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = \
        [(Jneg_excit2excit +
          (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
         for j in range(N_excitatory)]

    # Add noise to the synaptic weights
    presyn_weight_kernel = add_gaussian_white_noise_by_magnitude(presyn_weight_kernel, synaptic_noise_amount)

    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

    # define the inhibitory population
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - G_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_lif_dynamics,
        threshold="v>v_firing_threshold_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1.0",
                            refractory=t_abs_refract_excit, method="rk2")
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               on_pre="s_AMPA += 1.0", method="rk2")
    syn_excit2inhib.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop,
    #        model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)

    # set the STRUCTURED recurrent input. use a network_operation
    @network_operation()
    def update_nmda_sum():
        fft_s_NMDA = rfft(excit_pop.s_NMDA)
        fft_s_NMDA_total = numpy.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
        s_NMDA_tot = irfft(fft_s_NMDA_total)
        excit_pop.s_NMDA_total_ = s_NMDA_tot

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = distractor_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        idx_monitored_neurons = \
            [int(math.ceil(k))
             for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)

    b2.run(sim_time)
    return \
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45



def simulate_wm_reduced2(
    N_excitatory=1024, N_inhibitory=256,
    N_extern_poisson=1000, poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
    sigma_weight_profile=20., Jpos_excit2excit=1.6,
    stimulus_center_deg=180, stimulus_width_deg=40, stimulus_strength=0.07 * b2.namp,
    t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
    distractor_center_deg=90, distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
    t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
    G_inhib2inhib=.35 * 1.024 * b2.nS,
    G_inhib2excit=.35 * 1.336 * b2.nS,
    G_excit2excit=.35 * 0.381 * b2.nS,
    G_excit2inhib=.35 * 1.2 * 0.292 * b2.nS,
    G_extern2inhib = 2.38 * b2.nS, # Temp for testing
    G_extern2excit = 3.1 * b2.nS, # Temp for testing
    monitored_subset_size=1024, sim_time=800. * b2.ms,
    synaptic_noise_amount=0.0):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +\- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time
    Returns:
       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    # specify the excitatory pyramidal cells:
    Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 25.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 2.0 * b2.ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction

    #
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib = 0.2 * b2.nF
    G_leak_inhib = 20.0 * b2.nS
    E_leak_inhib = -70.0 * b2.mV
    v_firing_threshold_inhib = -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 1.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = .9 * 2.0 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = .65 * 100.0 * b2.ms  # orig: 100
    tau_NMDA_x = .94 * 2.0 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    #G_extern2inhib = 2.38 * b2.nS # Temp removed
    #G_extern2excit = 3.1 * b2.nS # Temp removed

    # projectsions from the inhibitory populations
    G_inhib2inhib *= weight_scaling_factor
    G_inhib2excit *= weight_scaling_factor

    # projections from the excitatory population
    G_excit2excit *= weight_scaling_factor
    G_excit2inhib *= weight_scaling_factor  # todo: verify this scaling

    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = t_distractor_start + t_distractor_duration
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx,
                                                            distr_center_idx + distr_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = \
        [(Jneg_excit2excit +
          (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
         for j in range(N_excitatory)]

    # Add noise to the synaptic weights
    presyn_weight_kernel = add_gaussian_white_noise_by_magnitude(presyn_weight_kernel, synaptic_noise_amount)

    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

    # define the inhibitory population
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - G_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_lif_dynamics,
        threshold="v>v_firing_threshold_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1.0",
                            refractory=t_abs_refract_excit, method="rk2")
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               on_pre="s_AMPA += 1.0", method="rk2")
    syn_excit2inhib.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop, model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)
    syn_excit2excit = Synapses(excit_pop, excit_pop, 'w : 1', on_pre="s_AMPA += w")
    #syn_excit2excit.connect(condition="i!=j", p=1.)
    syn_excit2excit.connect(p=1.)
    excit_pop.x = 'i' # Set an index to each neuron
    Gain = Jpos_excit2excit
    Gain = Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit)
    syn_excit2excit.w = 'Gain * exp( -(360.0 * min(abs(x_pre-x_post), N_excitatory - abs(x_pre-x_post)) / N_excitatory)**2 / (2 * sigma_weight_profile**2))'

    # # set the STRUCTURED recurrent input. use a network_operation
    # @network_operation()
    # def update_nmda_sum():
    #     fft_s_NMDA = rfft(excit_pop.s_NMDA)
    #     fft_s_NMDA_total = numpy.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
    #     s_NMDA_tot = irfft(fft_s_NMDA_total)
    #     excit_pop.s_NMDA_total_ = s_NMDA_tot

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = distractor_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        idx_monitored_neurons = \
            [int(math.ceil(k))
             for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)

    b2.run(sim_time)
    return \
        rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45





def getting_started():
    b2.defaultclock.dt = 0.1 * b2.ms
    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile\
        = simulate_wm(N_excitatory=256, N_inhibitory=64, weight_scaling_factor=8., sim_time=500. * b2.ms,
                      stimulus_center_deg=120, t_stimulus_start=100 * b2.ms, t_stimulus_duration=200 * b2.ms,
                      stimulus_strength=.07 * b2.namp)
    plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit,
                                     t_min=0. * b2.ms)
    plt.show()

if __name__ == "__main__":
    getting_started()
