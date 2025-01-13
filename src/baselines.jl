# defines baseline models for comparision 

# QG3 

using EllipsisNotation, OrdinaryDiffEq, SciMLSensitivity

"""
    model = QG3Baseline(m::QG3Model; kwargs...)

Predictor baseline, just the QG3 model directly
"""
function QG3Baseline(m::QG3Model{T}; alg=Tsit5(), kwargs...) where T
    GPU = QG3.isongpu(m)
    S, __, __, q_0 = load_data("T42", GPU=GPU)

    rhs = WrappedFunction(x -> QG3.QG3MM_gpu(x, [m, S], 0f0))
    model = SciMLNeuralDE(rhs; alg=alg, kwargs...)

    return model
end 

"""
    model = JBaseline(m::QG3Model; kwargs...)

Predictor baseline, just the Jacobian directly
"""
function JBaseline(m::QG3Model{T}; kwargs...) where T
    GPU = QG3.isongpu(m)
    __, __, __, q_0 = load_data("T42", GPU=GPU)

    rhs = WrappedFunction(x -> QG3.QG3MM_adv_gpu(x, m, 0f0))
    return SciMLNeuralDE(model=rhs, alg=Tsit5(), kwargs...)
end 

"""
    RecursiveNet{M}

Resurively applied ANN. Can be called like [`ChaoticNDE`](@ref) with a tuple `(t, x)`

Regular input assumes WxHxC data 
"""
struct RecursiveNet{M} <: Lux.AbstractLuxWrapperLayer{:nn}
    nn::M
end 

(m::RecursiveNet)(x,ps,st) = m.nn(x, ps, st)

ConvBaseline(; kwargs...) = RecursiveNet(UNet(; kwargs...))

"""
    load_node_unet(save_name, qg3p::QG3Model, DT::Number; save_dir="", rng=Random.default_rng(), dev=DetermineDevice(), kwargs...)

Load an already trained UNet NODE model from `save_name` file (dq/dt = S(UNet(S^{-1}(q)))).
"""
function load_node_unet(save_name, qg3p::QG3Model, DT::Number; save_dir="", rng=Random.default_rng(), dev=DetermineDevice(), kwargs...)
    (; SAVE_NAME, SAVE_NAME_MODEL) = NeuralQG3.setup_savenames(default_name=save_name, save_dir=save_dir)
    @load SAVE_NAME_MODEL psn_hyperpars

    nn_single = FlatChain(NeuralQG3.TransformGridLayer(qg3p.g.SHtoG), WrappedFunction(x->reshape(x,size(x)...,1)),NeuralQG3.PermutedimLayer((2,3,1,4)), NeuralQG3.UNet(; psn_hyperpars.pars[:unet_kwargs]..., activation=psn_hyperpars.pars[:activation]), NeuralQG3.PermutedimLayer((3,1,2,4)), WrappedFunction(x->view(x,:,:,:,1)), NeuralQG3.TransformSHLayer(qg3p.g.GtoSH))
    neural_de_sciml = SciMLNeuralDE(model=nn_single, alg=Tsit5(), sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true), dt=DT, kwargs...)

    ps, st = Lux.setup(rng, neural_de_sciml)
    ps = load_ps(SAVE_NAME, dev)

    return neural_de_sciml, ps, st
end 

"""
    load_rnn_unet(save_name; save_dir="", rng=Random.default_rng(), dev=DetermineDevice())

Load an already trained UNet RNN model from `save_name` file.
"""
function load_rnn_unet(save_name; save_dir="", rng=Random.default_rng(), dev=DetermineDevice())
    (; SAVE_NAME, SAVE_NAME_MODEL) = NeuralQG3.setup_savenames(default_name=save_name, save_dir=save_dir)
    @load SAVE_NAME_MODEL psn_hyperpars

    nn = FlatChain(NeuralQG3.PermutedimLayer((2,3,1,4)), NeuralQG3.UNet(; psn_hyperpars.pars[:unet_kwargs]..., activation=psn_hyperpars.pars[:activation]), NeuralQG3.PermutedimLayer((3,1,2,4)))
    neural_de = NeuralQG3.RecursiveNet(nn)
    ps, st = Lux.setup(rng, neural_de)
    ps = load_ps(SAVE_NAME, dev)
    
    return neural_de, ps, st
end 
    