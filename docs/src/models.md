# Models 

Building on the previously defined layers, we can set up the complete PseudoSpectralNet. For this purpose we have different routines based on how we build up the model and specify its hyperparameters. 

```@docs; canonical=false
NeuralQG3.PseudoSpectralNet
```

which is wrapped inside a `PSN_RHS` and the `NeuralDELux.ADNeuralDE` and `NeuralDELux.SciMLNeuralDE` container formats to set up a Neural DE 

```@docs; canonical=false
NeuralQG3.PSN_RHS
NeuralDELux.ADNeuralDE
NeuralDELux.SciMLNeuralDE
```

Additionally we have to routines that make it easier to load trained models. Those routines directly return instances of `NeuralDELux.ADNeuralDE` and `NeuralDELux.SciMLNeuralDE` that are Lux' layers to act as Neural DEs. 

```@docs; canonical=false
load_psn_ad
load_psn_sciml
```
