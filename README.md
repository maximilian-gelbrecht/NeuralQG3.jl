![NeuralQG3 Logo](logo.png)

# NeuralQG3.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://maximilian-gelbrecht.github.io/NeuralQG3.jl/dev/)
[![Build Status](https://github.com/maximilian-gelbrecht/NeuralQG3.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/maximilian-gelbrecht/NeuralQG3.jl/actions/workflows/CI.yml?query=branch%3Amain)

Here, [QG3.jl](https://github.com/maximilian-gelbrecht/QG3.jl.git) gets the Neural PDE treatment. All of the details and the method are outlined in the paper currently under review. Please refer to that paper for all details, here we will just present ways how to use the code in the documentation. 

The physics-based core of the PseudoSpectralNet is taken directly from the implemenation of the quasigeostrophic model in [QG3.jl](https://github.com/maximilian-gelbrecht/QG3.jl.git) is not part of this repository. Here, we only implement Lux.jl layers that call those routines. 

## Installation 

This package depends on several unregistered packages. Julia 1.11 should be able to instantiate those as their URLs are given in the respective `Project.toml` files. However, there are currently multiple problems with this. If this isn't working or you are using an older Julia version, please inspect the `[sources]` section of `Project.toml` and add those unregistered packages manually. 

In order to run the Speedy experiment with the same data, we used, you can download it from [here (with password "speedy-data")](https://cloud.pik-potsdam.de/index.php/s/g43p23Eoza8sTf8) and put in the `data-files/speedy` folder. 

## Comment about dependencies and code status 

Several of the dependencies of this project have been undergoing mayor changes since the inception of the PseudoSpectralNet / NeuralQG3.jl work. In this repo we tried to keep compability with the updated versions of all dependencies at least to a certain degree though. This is e.g. the reason we use a custom `Chain` layer that restores the behaviour of the `Chain` layer in older Lux version. Simarly, we trained our model on older versions of SpeedyWeather.jl. We saved the respective dataset in the LFS of the repository. 

Unfortunately some of the dependencies of the project stopped development (e.g. Tullio.jl). Further updating the environment might break some scripts. We will try to fix those things as long as they are easy to fix, but most of our efforts are being spend to apply similar ideas in a new project to `SpeedyWeather.jl`.

## Cite us 

If you are using this model for any publications or other works, please cite us. 

- ZENODO and ARXIV link upcoming (please get in touch with us) - 
