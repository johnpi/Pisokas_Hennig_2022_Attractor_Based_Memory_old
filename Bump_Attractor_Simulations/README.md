# Attractor_Based_Memory_Plaussibility_Study


The realisation that a line or ring attractor activity packet or bump would drift in specific way leads to a method to test and falsify the hypothesis that the PI distance memory is employing a ring atractor circuit. 

There are multiple types of noise in an attractor circuit. An attractor will drift in a manner dependend on the input noise --- if it includes shifting circuitry --- and the type of heterogeneity in the attractor synaptic strengths and neuron biophysical properties. If the synaptic strengths have Gaussian white noise and there are many neurons the drift will be in one direction with constant rate in respect to time. If the synaptic strengths variations are not evenly distributed but rather some synapses might be stronger some weaker without multiple neurons averaging out these local variations then the activity bump will driftwith rate that depends in its location along the attractor. Drift can even change direction or even seem to be almost stationary if two neighbouring synaptic variations push it towards each other. 

The mathematical analysis of a ring attractor circuit supports these claims. Before exploring the circuit analytically I am running some simulations as quick test of these observations.
We implement this as ring-attractor to avoid the complication of boundary conditions, that is dealing with the edges of the network. This simplification does not affect the conclusions of the current study. Whether the neural implementation in the animal is a ring or line attractor will affect what will happen if we force the animal to exceed its maximum distance that can be represented by the network. Such an experiment is in the plan but not done yet. 

There are two implementations, first a rate based one and second a spiking neural implementation. 

The drift regime of the attractors will be compared with the memory drift derived by behavioural experiments and compared. If the memory decays not linearly with time or it does not fluctuate then the substrate of PI memory cannot an attractor circuit or at least not of the vanilla type suggested in the literature (Skaggs 1995). 


## Files:

README.md This file.

neurodynex/ Contains the the libraries from the Neuronal Dynamics book from which I am using the model of ring attractor. It also includes my modified version of the ring attractor used in the various experiments here. 

Data/ 

merge_all_files_into_one_keep_only_thetas.py Process the collected data files (.npy) extract the theta time series from the population activity and store only the theta and some other useful data disregarding the individual neuronal activity to save space. 

Paper_experiments-Copy1.ipynb The plots for the paper. 

line-attractor-spiking-neurodynexlib-simplified-neurons-ECLV-principal-neurons-altered-Copy1.ipynb One of the files with temporal experiments with the ring attractor parameters. 

run_trials_job.sh The script starting the data collection job. It is made for running on the Eddie grid but can also be run on other machines. 


Other files to commend on: 
-rw-r--r--@  1 john  staff    28678 22 Mar 20:10 optimize_synaptic_conductances.py
-rw-r--r--@  1 john  staff    19809 22 Mar 21:51 run_trials-simplified-neurons_EC_LV_Principal_Neurons_1.py
-rwxr-xr-x@  1 john  staff     3713 22 Mar 21:53 run_trials.sh
-rw-r--r--@  1 john  staff     3548 26 May 22:54 merge_all_files_into_one.py
-rw-r--r--@  1 john  staff      960 26 May 22:54 print_elements_num.py
-rwxr-xr-x@  1 john  staff      484 26 May 23:00 print-trials-collected.sh
-rwxr-xr-x@  1 john  staff     1195 26 May 23:06 backup-new-files.sh
-rwxr-xr-x@  1 john  staff     1350 26 May 23:06 collect_more_auto.sh
-rwxr-xr-x@  1 john  staff     1803 26 May 23:06 move-completed-to-store.sh
-rwxr-xr-x@  1 john  staff     2113 26 May 23:06 recover-failed.sh
-rw-r--r--@  1 john  staff     1061 26 May 23:06 restart-failed.sh
-rwxr-xr-x@  1 john  staff      104 26 May 23:06 show-running.sh
-rwxr-xr-x@  1 john  staff      105 26 May 23:06 show-waiting.sh
-rwxr-xr-x@  1 john  staff      125 26 May 23:06 tasks-running-params.sh
-rw-r--r--@  1 john  staff     4589 28 Nov  2020 correct_num_of_samples.py
-rw-r--r--@  1 john  staff    29181  5 Dec  2020 utility_functions.py
-rw-r--r--   1 john  staff    21890  5 Dec  2020 utility_functions.pyc
-rwxr-xr-x@  1 john  staff      790  6 Dec  2020 watch_script_warn_email.sh
-rw-r--r--@  1 john  staff    15001  6 Feb  2020 run_trials.py



## Examples of how to collect and process the data for different network sizes
### Count the number of trials in each of the files and save it in files output-new-${n}.txt
> for n in 128 256 512 1024 2048 4096 8192; do echo "N=$n..."; for f in /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/Data/*_${n}.npy; do  python3 print_elements_num.py -f "${f}"; done > /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/output-new-${n}.txt ; done 
N=128...
N=256...
N=512...
N=1024...
N=2048...
N=4096...
N=8192...

### Show how many files do we have
> for n in 128 256 512 1024 2048 4096 8192; do echo "N=$n... $(ls -l /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/Data/*_$n.npy | wc -l) $(cat /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/output-new-$n.txt 2> /dev/null | wc -l)"; done
N=128...       46       46
N=256...      195      195
N=512...      243      243
N=1024...      220      220
N=2048...      290      290
N=4096...      540      540
N=8192...      945      945

### Merge the output files into one
> rm /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/output-new-all.txt
> cat /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/output-new-*.txt > /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/output-new-all.txt

### Display how many trials we got for each condition
> ./print-trials-collected.sh /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/output-new-all.txt

NMDA
--------
N=128 noise=0.001 trials=108
N=128 noise=0.002 trials=147
N=128 noise=0.003 trials=100
N=128 noise=0.004 trials=100
N=128 noise=0.005 trials=164
N=128 noise=0.006 trials=100
N=128 noise=0.007 trials=100
N=128 noise=0.008 trials=135
N=128 noise=0.009 trials=114

N=256 noise=0.001 trials=256
N=256 noise=0.002 trials=143
N=256 noise=0.003 trials=183
N=256 noise=0.004 trials=159
N=256 noise=0.005 trials=249
N=256 noise=0.006 trials=165
N=256 noise=0.007 trials=129
N=256 noise=0.008 trials=157
N=256 noise=0.009 trials=147

...

EC_LV_1
--------
N=128 noise=0.001 trials=101
N=128 noise=0.002 trials=101
N=128 noise=0.003 trials=101
N=128 noise=0.004 trials=101
N=128 noise=0.005 trials=139
N=128 noise=0.006 trials=101
N=128 noise=0.007 trials=101
N=128 noise=0.008 trials=101
N=128 noise=0.009 trials=101

N=256 noise=0.001 trials=117
N=256 noise=0.002 trials=165
N=256 noise=0.003 trials=112
N=256 noise=0.004 trials=115
N=256 noise=0.005 trials=138
N=256 noise=0.006 trials=111
N=256 noise=0.007 trials=137
N=256 noise=0.008 trials=110
N=256 noise=0.009 trials=140

...

### Extract the population vector theta time series and store them in one file collected-wrapped.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 1 -i /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/Data/ -o /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/collected-wrapped.npy

### Extract the population vector theta time series, unwrap them, and store them in one file collected-unwrapped.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --unwrap-angles -i /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/Data/ -o /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/collected-unwrapped.npy



## Example of how to collect and process data for different membrane time constant values

### The NMDA model with tau modification with the New way of manipulating tau now it is the capacitance I change instead of the conductance.
> for n in 1.4; do s=256; N=10; for tau in 0.5 1 5 10 20 30 40 50 60 70 80 90 100; do for i in `seq 100 450`; do qsub ./run_trials_job_64G.sh NMDA-TAU $s 300 $N $n eddie$i $tau; done; done; done

### Display the number of files collected for each condition
> N=256; ORR=$(./show-running.sh); OWW=$(./show-waiting.sh); for n in 1.4; do echo "-----------------"; echo "Neurons $N"; echo "Noise $n"; echo "tau waiting running files"; echo "-----------------"; for t in 0.5 1 5 10 20 30 40 50 60 70 80 90 100; do echo "${t}ms $(echo "${OWW}" | grep " $t\$" | grep " $n " | grep " $N " | wc -l) $(echo "${ORR}" | grep " $t\$" | grep " $n " | grep " $N " | wc -l) $(ls -l /exports/eddie/scratch/s0093128/Data/Backup/*_tau${t}ms_*  2> /dev/null | grep "${n}Hz" | grep "_$N" | wc -l)"; done; done

### Useful commands for monitoring collection progress
> echo "qw=$(qstat | grep ' qw ' | wc -l) r=$( qstat | grep ' r ' | wc -l)"

### Useful commands for deleting files that have been downloaded to the local computer
files=()
for f in ${files[@]}; do rm Data/$f; rm /exports/eddie/scratch/s0093128/Data/Backup/$f; rm /exports/eddie/scratch/s0093128/Data/${f/.gz/}; done


### Extract the population vector theta time series, and store them in one file collected-wrapped-NMDA-TAU.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 2 -i /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/Data/ -o /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/collected-wrapped-NMDA-TAU.npy

### Extract the population vector theta time series, unwrap them, and store them in one file collected-unwrapped-NMDA-TAU.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 2 --unwrap-angles -i /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/Data/ -o /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/collected-unwrapped-NMDA-TAU.npy


## Collect data with systematic bump shifting

### Collect only the NMDA-SHIFT-0.001 and NMDA-SHIFT-0.0005 for 300s (--weights-skewness 0.05 results in 180deg shift per ~8.6s (20.960083234484586deg/s))

> NMDA_SHIFTING=-0.0005; for n in 0.005; do s=256; N=5; for i in `seq 100 300`; do qsub ./run_trials_job_64G.sh NMDA-SHIFT $s 300 $N $n eddie$i 0 ${NMDA_SHIFTING}; done; done
> NMDA_SHIFTING=-0.001; for n in 0.005; do s=256; N=5; for i in `seq 100 130`; do qsub ./run_trials_job_64G.sh NMDA-SHIFT $s 300 $N $n eddie$i 0 ${NMDA_SHIFTING}; done; done

### Display the number of files collected for each condition
> N=256; ORR=$(./show-running.sh); OWW=$(./show-waiting.sh); for n in 0.005; do echo "-----------------"; echo "Neurons $N"; echo "Noise $n"; echo "NMDA-SHIFT waiting running files"; echo "-----------------"; for s in -0.0005 -0.001; do echo "${s} $(echo "${OWW}" | grep "NMDA-SHIFT" | grep " $s\$" | grep " $n " | grep " $N " | wc -l) $(echo "${ORR}" | grep "NMDA-SHIFT" | grep " $s\$" | grep " $n " | grep " $N " | wc -l) $(ls -l /exports/eddie/scratch/s0093128/Data/Backup/*_NMDA-SHIFT${s}_*  2> /dev/null | grep "${n}Hz" | grep "_$N" | wc -l)"; done; done

### Extract the population vector theta time series, and store them in one file collected-wrapped-NMDA-SHIFT.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 3 -i /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/Data/ -o /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/collected-wrapped-NMDA-SHIFT.npy

### Extract the population vector theta time series, unwrap them, and store them in one file collected-unwrapped-NMDA-SHIFT.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 3 --unwrap-angles -i /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/Data/ -o /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/collected-unwrapped-NMDA-SHIFT.npy


