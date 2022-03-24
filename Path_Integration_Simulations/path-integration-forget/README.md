# Central complex path integration model

The code in this directory is derived from the code created by Thomas Stone and published with the paper Stone, T. et. al. (2017). An anatomically constrained model for path integration in the bee brain. Current Biology, 27(20), 3069-3085.


Parameters are specified in central_complex.py



## How to use

1. Run the script ```collect_simulation_path_data.py``` as shown in the examples listed in the comments to collect simulation data. To simplify usage I have included a pregenerated outbound trip recording (data/outbound_route_only_NW_to_SE_1500steps_03.npz) that the examples use. 

2. Follow the examples shown in ```collect_simulation_path_data.py``` to organise the generated files in directories. 

3. Use the script ```npz_to_csv.ipynb``` convert the .npz generated with the previous steps to .csv files. 



