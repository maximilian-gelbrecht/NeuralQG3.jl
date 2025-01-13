# Baselines 

The baselines used in the paper can be initialized with the help of the following routines

```@docs 

NeuralQG3.QG3Baseline
NeuralQG3.JBaseline
NeuralQG3.RecursiveNet
```

These use our UNet implementation

```@docs; canonical=false 
NeuralQG3.UNet
NeuralQG3.ConvBlock
```

and trained models can be loaded with the helper routines 

```@docs; canonical=false 
NeuralQG3.load_node_unet
NeuralQG3.load_rnn_unet
```