# Hyperparameters 

The hyperparemters of the PseudoSpectralNet are saved in a struct 

```@docs; canonical=false
NeuralQG3.PSNHyperparameters
```

which can be setup by the routine `parse_pars`

```@docs; canonical=false
NeuralQG3.parse_pars
```

The training/optimization specific parameters can also be directly be initialised via

```@docs; canonical=false
NeuralQG3.training_hyperpars
```

and the file names used during I/O are constructed by 

```@docs; canonical=false
NeuralQG3.setup_savenames
```