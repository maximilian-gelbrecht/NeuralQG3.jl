# Data 

This folder contains data files needed to use PseudoSpectralNet and scripts to reproduce the data used in the paper. 

## Precomputed QG3 Files 

The JLD2 files present in this folder are precomputed fields for QG3.jl in a T42 resolution. They are used when calling `NeuralQG3.load_data("T42"). 

## SpeedyWeather.jl scripts 

In the paper we train PseudoSpectralNet on data from SpeedyWeather.jl's primitive equation dry core. The model is still under development and changes to its dynamics were introduced after our work on this project began. While the main project and script enviroment use an more up-to-date version of SpeedyWeather.jl to avoid any updating conflicts with other packages, here we provide scripts to produce exactly the kind of data that is also used in the paper with SpeedyWeather.jl v0.7.1. For this purpose make sure you activate the environment and instantiate it. Alternatively you can also download the data we used at [here (with password "speedy-data")](https://cloud.pik-potsdam.de/index.php/s/g43p23Eoza8sTf8). 

The script `generate_speedy_data.jl` saves a voritcity trajectory in a JLD2 file that can be loaded later with `get_data(:speedy, qg3p::QG3Model, false; speedy_file_name="data-files/speedy/speedy-qt-precomputed.jld2")` or just by directly using JLD2.jl. 

