# Utilities 

## Data loaders and generation 

The PSN model uses the process-based compoments from QG3.jl. Those need to be pre-computed, the repository includes those for a T42 grid, which can be loaded via 

```@docs; canonical=false
NeuralQG3.load_data
```

The actual training data can be computed via 

```@docs; canonical=false
NeuralQG3.get_data
```

which calls 

```@docs; canonical=false
NeuralQG3.compute_QG3_data
NeuralQG3.compute_speedy_data
NeuralQG3.compute_ERA_data
```

and uses the [NODEData.jl](https://github.com/maximilian-gelbrecht/NODEData.jl) package for dataloaders. 