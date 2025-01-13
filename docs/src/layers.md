# Layers 

The core of PseudoSpectralNet (PSN) are the layers that inform PSN about the process-based based core. As outlined in our paper these are seperated both by whether they receive input in grid space ($\mathcal{K}(\mathbf{x})$) or spherical harmonics space ($\mathca{K}_{lm}$) and whether or not the computation is performed dynamically at each time step or pre-computed once while setting up the model. The actual physical computations are imported from QG3.jl.

NeuralQG3.jl realises those layers as 

```@docs; canonical=false
NeuralQG3.KnowledgeGridLayer
NeuralQG3.KnowledgeSHLayer
```

Additionally, we define seperate layers just for the transform: 

```@docs; canonical=false
NeuralQG3.TransformSHLayer
NeuralQG3.TransformGridLayer
```