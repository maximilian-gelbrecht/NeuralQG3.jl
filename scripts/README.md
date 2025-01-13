# Scripts 

This folder contains all scripts to run the experiments of the main article, evaluate the experiments and plot the results. 

The key scripts are: 

* `run_pseudospectralnet-hyperpar.jl`: Train a PSN on QG3 data, can also be used for a hyperparameter optimiziation 
* `run_pseudospectralnet-speedy.jl`: Train a PSN on SpeedyWeather.jl data 
* `run_pseudospectralnet-ERA.jl`: Train a PSN on ERA5 data
* `run_pseudospectralnet-baseline.jl`: Train the NODE SH baseline 
* `run_unet_RNN.jl`: Train the RNN baseline 
* `eval_psn.ipynb`: Main evaluation Jupyter notebook, produces plots for forecasts 
* `eval_stability_*.jl`: Evaluate the long-term stablity of the trained models 
* `eval_stability_energy*.jl`: Evaluate the long-term kinetic energy of the trained models for stability tests