# Attractor_Based_Memory_Plaussibility_Study


The realisation that a line or ring attractor activity packet or bump would drift in specific way leads to a method to test and falsify the hypothesis that the PI distance memory is employing a ring atractor circuit. 

There are multiple types of noise in an attractor circuit. An attractor will drift in a manner dependend on the input noise --- if it includes shifting circuitry --- and the type of heterogeneity in the attractor synaptic strengths and neuron biophysical properties. If the synaptic strengths have Gaussian white noise and there are many neurons the drift will be in one direction with constant rate in respect to time. If the synaptic strengths variations are not evenly distributed but rather some synapses might be stronger some weaker without multiple neurons averaging out these local variations then the activity bump will driftwith rate that depends in its location along the attractor. Drift can even change direction or even seem to be almost stationary if two neighbouring synaptic variations push it towards each other. 

The mathematical analysis of a ring attractor circuit supports these claims. Before exploring the circuit analytically I am running some simulations as quick test of these observations.
We implement this as ring-attractor to avoid the complication of boundary conditions, that is dealing with the edges of the network. This simplification does not affect the conclusions of the current study. Whether the neural implementation in the animal is a ring or line attractor will affect what will happen if we force the animal to exceed its maximum distance that can be represented by the network. Such an experiment is in the plan but not done yet. 

There are two implementations, first a rate based one and second a spiking neural implementation. 

The drift regime of the attractors will be compared with the memory drift derived by behavioural experiments and compared. If the memory decays not linearly with time or it does not fluctuate then the substrate of PI memory cannot an attractor circuit or at least not of the vanilla type suggested in the literature (Skaggs 1995). 

