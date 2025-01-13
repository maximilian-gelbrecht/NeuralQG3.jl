module NeuralQG3

    using QG3, CUDA, Adapt
    using Lux, Zygote, Printf, JLD2, Random
    using Tullio, KernelAbstractions, SpeedyWeather, Plots, DocStringExtensions

    import StatsBase
    import Base.show, Base.summary
    import Optimisers.trainable

    const Ω = 7.2921159f-5
    const earth_radius = 6.371f6

    include("gpu.jl")
    include("hyperpars.jl")
    include("utility-layers.jl")
    include("utility-funcs.jl")
    include("qg_data.jl")
    include("pseudoconv.jl")
    include("psn-layers.jl")
    include("speedy-data.jl")
    include("loss.jl")
    include("unet.jl")
    include("baselines.jl")

    export PseudoSpectralNetQG3, PseudoSpectralNetQG3Add, PseudoSpectralNetSW, PseudoSpectralNet, PSNHyperparameters
    export load_data

end 