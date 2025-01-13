"""
    KnowledgeGridLayer(p::QG3Model{T}; additional_knowledge=false) <: Lux.AbstractLuxLayer

Knowledge layer in the grid space that uses QG3.jl to compute process-based knowledge. Receive input in the SPH space that is immediatly transformed into the grid domain. 
It holds the pre-computed knowledge of the grid domain and comptues the online knowledge (derivates etc). Concats everything along dim 1.

The flag `additional_knowledge` toggles whether or not all components of the QG3 model are used or only a reduced version (as in the QG3 application example in the paper). 

In the current form this layer has no trainable parameters, however, it would be reasonble simple to make all physical parameters trainable as well. If you interested in that, do a GitHub issue. 

It is a subtype of `Lux.AbstractLuxLayer` and is therefore called as any other Lux layer with three arguments `(x, ps, st)`. 
"""
struct KnowledgeGridLayer{T,DM,DL,SH,DD,DN,F1,F2,S} <: Lux.AbstractLuxLayer
    knowledge::T 
    dμ::DM 
    dλ::DL 
    SH2G::SH
    L::DD 
    ∇8::DN 
    ∂f_J3∂λ::F1 # derivate coriolis drag from QG3 model used for Jacobian
    ∂f_J3∂μ::F2 # derivate coriolis drag from QG3 model used for Jacobian
    t_SH::S
end 

function KnowledgeGridLayer(p::QG3Model{T}; additional_knowledge=false, N_batch::Int=0) where {T}
    cosϕ = zeros(T,1,p.p.N_lats,p.p.N_lons)
    μ = zeros(T,1,p.p.N_lats,p.p.N_lons)
    g = p.g
    for ilat ∈ 1:p.p.N_lats
        cosϕ[1,ilat,:] .= cos(p.p.lats[ilat])
        μ[1,ilat,:] .= p.p.μ[ilat]
    end

    k_SH = QG3.transform_SH(p.k, p)

    if !(additional_knowledge) 
        knowledge = cat(reshape(p.k,1,p.p.N_lats,p.p.N_lons), # drag, k
        reshape(QG3.SHtoGrid_dμ(k_SH,p), 1,p.p.N_lats,p.p.N_lons),   # ∂k∂μ
        reshape(QG3.SHtoGrid_dλ(k_SH,p), 1,p.p.N_lats,p.p.N_lons),   # ∂k∂λ
        reshape(QG3.transform_grid(QG3.Δ(k_SH, p), p), 1,p.p.N_lats,p.p.N_lons),   # Δk
        cosϕ,    # cosϕ
        μ,    # sinϕ/μ
        dims=1)
    else additional_knowledge 
        h_SH = transform_SH(p.p.h ./p.p.H0, p)

        knowledge = cat(reshape(p.k,1,p.p.N_lats,p.p.N_lons), # drag, k
        reshape(QG3.SHtoGrid_dμ(k_SH,p), 1, p.p.N_lats,p.p.N_lons),   # ∂k∂μ
        reshape(QG3.SHtoGrid_dλ(k_SH,p), 1, p.p.N_lats,p.p.N_lons),   # ∂k∂λ
        reshape(QG3.transform_grid(QG3.Δ(k_SH, g), p), 1,p.p.N_lats,p.p.N_lons),   # Δk
        reshape(p.p.h/p.p.H0, 1,p.p.N_lats,p.p.N_lons), # h 
        reshape(p.p.LS, 1,p.p.N_lats,p.p.N_lons), # LS 
        reshape(QG3.SHtoGrid_dμ(h_SH,p), 1, p.p.N_lats,p.p.N_lons),   # ∂h∂μ
        reshape(QG3.SHtoGrid_dλ(h_SH,p), 1, p.p.N_lats,p.p.N_lons),   # ∂h∂λ
        cosϕ,    # cosϕ
        μ,    # sinϕ/μ
        dims=1)
    end

    ∂f_J3∂λ = QG3.SHtoGrid_dλ(p.f_J3, p)
    ∂f_J3∂μ = QG3.SHtoGrid_dμ(p.f_J3, p)        

    extra_transform = QG3.SHtoGaussianGridTransform(p.p, 3*6; N_batch=N_batch) # during the pass through there are 6 fields with 3 levels each
    grid_6 = QG3.grid(p.p, "gaussian", 6; N_batch=N_batch)

    dμ = grid_6.dμ
    dλ = grid_6.dλ 
    sh2g = grid_6.SHtoG
    Δ = grid_6.Δ 
    ∇8 = grid_6.∇8 

    if N_batch > 0 
        knowledge = repeat(knowledge, 1,1,1,N_batch)
        ∂f_J3∂λ = repeat(∂f_J3∂λ, 1,1,1,1)
        ∂f_J3∂μ = repeat(∂f_J3∂μ, 1,1,1,1)
    end
    
    return KnowledgeGridLayer{typeof(knowledge), typeof(dμ), typeof(dλ), typeof(sh2g), typeof(Δ), typeof(∇8), typeof(∂f_J3∂λ), typeof(∂f_J3∂μ), typeof(extra_transform)}(knowledge, dμ, dλ, sh2g, Δ, ∇8, ∂f_J3∂λ, ∂f_J3∂μ, extra_transform)
end

function (m::KnowledgeGridLayer)(x, ps, st) # this is more like KnowledgeLayerSHtoGrid 

    ∂f_J3∂μ = m.∂f_J3∂μ
    ∂f_J3∂λ = m.∂f_J3∂λ
    t_SH = m.t_SH

    ∂x∂μ = QG3.SHtoGrid_dμ(x, m.dμ)
    ∂x∂λ = QG3.SHtoGrid_dλ(x, m.dλ, m.SH2G)

    J_from_pre = (selectdim(∂x∂μ,1,4:6) .+ ∂f_J3∂μ .- eltype(x)(1)) .* selectdim(∂x∂λ,1,1:3) - (selectdim(∂x∂λ,1,4:6) .+ ∂f_J3∂λ) .* selectdim(∂x∂μ,1,1:3)  
    
    return cat(QG3.transform_grid(cat(x, 
                                      QG3.Δ(x, m.L),
                                      QG3.cH∇8(x, m.∇8),
                                      dims=1), t_SH),
                ∂x∂μ, 
                ∂x∂λ,  
                J_from_pre,
                m.knowledge,
                dims=1), st
end

show(io::IO, m::KnowledgeGridLayer) = show(io::IO, "Knowledge Grid Layer")

"""
    KnowledgeSHLayer(p::QG3Model{T}; additional_knowledge=false, GPU=false, S=nothing) where {T} <: Lux.AbstractLuxLayer

Knowledge layer in the spectral space that uses QG3.jl to compute process-based knowledge. Receive input in the SPH space. It holds the pre-computed knowledge of the spectral domain and computes the online knowledge (derivates etc). Concats everything along dim 1.

The flag `additional_knowledge` toggles whether or not all components of the QG3 model are used or only a reduced version (as in the QG3 application example in the paper). The flag `GPU` toggles GPU usage and `S` is used to hand over the forcing of the QG model that is not a part of `p` otherwise. 

In the current form this layer has no trainable parameters, however, it would be reasonble simple to make all physical parameters trainable as well. If you interested in that, do a GitHub issue. 

It is a subtype of `Lux.AbstractLuxLayer` and is therefore called as any other Lux layer with three arguments `(x, ps, st)`. 
"""
struct KnowledgeSHLayer{add,T,L,LA,H,TR} <: Lux.AbstractLuxLayer
    knowledge::T
    
    dλ::L    
    Δ::LA
    ∇8::H
    TR_matrix::TR
end 

function KnowledgeSHLayer(p::QG3Model{T}; additional_knowledge=false, GPU=false, S=nothing, N_batch::Int=0) where {T}
    g = p.g # grid info about derivative from QG3.jl parameters 
    
    l = QG3.lMatrix(p)
    m = QG3.mMatrix(p)
    
    if GPU 
         l = QG3.reorder_SH_gpu(l, p.p)
         m = QG3.reorder_SH_gpu(m, p.p)
    end
 
    if additional_knowledge
        pre = cat(reshape(l,(1,size(l)...)), reshape(m,(1,size(m)...)), S, dims=1)
    else 
        pre = cat(reshape(l,(1,size(l)...)), reshape(m,(1,size(m)...)), dims=1)
    end 
    
    g6 = QG3.grid(p.p, "gaussian", 6, N_batch=N_batch)

    TR_matrix = p.TR_matrix

    if N_batch > 0
        pre = repeat(pre, 1,1,1,N_batch)

        TR_matrix = repeat(TR_matrix, 1,1,1,N_batch)
        TR_matrix = reshape(TR_matrix, 3,3,:)
    end 

    return KnowledgeSHLayer{additional_knowledge, typeof(pre), typeof(g6.dλ), typeof(g6.Δ), typeof(g6.∇8), typeof(TR_matrix)}(pre, g6.dλ, g6.Δ, g6.∇8, TR_matrix)
end

function (m::KnowledgeSHLayer{false})(x, ps, st)
    
    return cat(x, 
                QG3.SHtoSH_dλ(x, m.dλ),
                QG3.Δ(x, m.Δ),
                QG3.∇8(x, m.∇8),
                m.knowledge,
                dims=1), st 
end 

function (m::KnowledgeSHLayer{true})(x, ps, st)

    return cat(x,
                reshape(batched_vec(m.TR_matrix, reshape(x[1:3,..],3,:)),3, size(x)[2:end]...),
                QG3.SHtoSH_dλ(x, m.dλ),
                QG3.Δ(x, m.Δ),
                QG3.∇8(x, m.∇8),
                m.knowledge,
                dims=1), st 
end

show(io::IO, m::KnowledgeSHLayer{true}) = show(io::IO, "Knowledge Layer SH w all processes")
show(io::IO, m::KnowledgeSHLayer{false}) = show(io::IO, "Knowledge Layer SH w reduced processes")


"""
    TransformSHLayer{T}(transform::QG3.SHtoGaussianGridTransform) <: Lux.AbstractLuxLayer

Wraps around a `QG3.SHtoGaussianGridTransform` to form a Lux layer that transforms the input from grid space to spectral space. 
"""
struct TransformSHLayer{T} <: Lux.AbstractLuxLayer
    transform::T
end 

(m::TransformSHLayer)(x, ps, st) = QG3.transform_SH(x, m.transform), st

"""
    TransformGridLayer{T}(transform::QG3.GaussianGridtoSHTransform) <: Lux.AbstractLuxLayer

Wraps around a `QG3.GaussianGridtoSHTransform` to form a Lux layer that transforms the input from spectral space to grid space. 
"""
struct TransformGridLayer{T} <: Lux.AbstractLuxLayer
    transform::T
end 

(m::TransformGridLayer)(x, ps, st) = QG3.transform_grid(x, m.transform), st

"""
PseudoSpectralNet{T} <: Lux.AbstractLuxWrapperLayer{:model}

PseudoSpectralNet Lux container layer.
"""
struct PseudoSpectralNet{T} <: Lux.AbstractLuxWrapperLayer{:model}
    model::T
end 

show(io::IO, m::PseudoSpectralNet) = show(io::IO, m.model)

"""
$(TYPEDSIGNATURES)
Initializes the PseudoSpectralNet. 
`nn_sh`, `nn_grid` and `nn_conv` are the three ANNs that are part of a PSN. `N_levels` sets the number of horizontal levels (default is 3). If `additional_knowledge==false` some components are of the QG3 are held back (for the QG3 appplication example). `S` is the forcing of the QG3 model. `ZNorm==true` includes a Z-normalization in the architecture. `N_batch` sets up the model to work with batches of this size. `N_batch==0` is for single sample use with the batch dimension squeezed (i.e. removed completely).   
`process_based` actually includes any knowledge based components, `process_based=false` can therefore be used as a baseline.
"""
function PseudoSpectralNet(nn_sh, nn_grid, nn_conv, p::QG3Model{T}; N_levels=3, additional_knowledge::Bool=false, process_based::Bool=true, GPU=false, S=nothing, znorm=true, conv_out_channels=3, conv_mode=:unet, unet_kwargs=NamedTuple(), scale_init_weight=Lux.ones32, scale_init_bias=Lux.zeros32, N_batch::Int=0, kwargs...) where T

    @assert conv_mode in [:unet,:pseudoconv] "conv_mode needs to be either :unet or :pseudoconv"
 
    # pre-compute the derivatives and co of the drag and prepare the other additional inputs 
    if process_based
        knowledge_grid_layer = KnowledgeGridLayer(p; additional_knowledge=additional_knowledge, N_batch=N_batch)
        knowledge_sh_layer = KnowledgeSHLayer(p; additional_knowledge=additional_knowledge, GPU=GPU, S=S, N_batch=N_batch)
    else # baseline, completley data-driven
        knowledge_grid_layer = TransformGridLayer(QG3.SHtoGaussianGridTransform(p.p, 6; N_batch=N_batch))
        knowledge_sh_layer = NoOpLayer()
    end 
    # prepare mask to zero out spurious coefficents 

    if N_batch > 0 
        SH_zero = QG3.SH_zero_mask(p.p, (N_levels,1,1,N_batch), N_batch=N_batch)
        SH_zero = GPU ? QG3.reorder_SH_gpu(SH_zero, p.p) : SH_zero
        SH_zero[:,1,1,:] .= 0 
    else 
        SH_zero = QG3.SH_zero_mask(p.p, (N_levels,1,1))
        SH_zero = GPU ? QG3.reorder_SH_gpu(SH_zero, p.p) : SH_zero
        SH_zero[:,1,1] .= 0 
    end

    # prepare in- and output sizes of the NNs 
    dummy_input = cat(QG3.zeros_SH(p; N_levels=N_levels, N_batch=N_batch), QG3.zeros_SH(p; N_levels=N_levels, N_batch=N_batch), dims=1)
    input_nn_grid, __ = knowledge_grid_layer(dummy_input, NamedTuple(), NamedTuple())
    input_knowledge_sh_grid, __ = knowledge_sh_layer(dummy_input, NamedTuple(), NamedTuple())

    nn_grid_inputsize = size(input_nn_grid, 1)
    nn_grid_outputsize = outputsize(nn_grid)

    # prepare the UNet (if we are using it)
    if conv_mode == :unet 
        if N_batch > 0 
            nn_conv = FlatChain(PermutedimLayer((2,3,1,4)), UNet(; in_channels=nn_grid_inputsize, out_channels=conv_out_channels, activation=kwargs[:activation], unet_kwargs...), PermutedimLayer((3,1,2,4))) # this is needed as the conv net needs WHCN format
        else 
            nn_conv = FlatChain(WrappedFunction(x->reshape(x,size(x)...,1)), PermutedimLayer((2,3,1,4)), UNet(; in_channels=nn_grid_inputsize, out_channels=conv_out_channels, activation=kwargs[:activation], unet_kwargs...), PermutedimLayer((3,1,2,4)), WrappedFunction(x->view(x,:,:,:,1))) # this is needed as the conv net needs WHCN format
        end
    end
    nn_conv_outputsize = conv_out_channels

    # additional normalization layers (data is already in normalized units)
    if znorm 
        znormsph = SPHZNormalize(p.p)
        znormgrid = GridZNormalize()
    else 
        znormsph = NoOpLayer()
        znormgrid = NoOpLayer()
    end 

    # define the NNs, including the reshapes for the Dense layers and the zeroing of spurious SH modes
    if N_batch > 0
        nn_grid = FlatChain(WrappedFunction(x->reshape(x, nn_grid_inputsize, :)), Dense(nn_grid_inputsize, inputsize(nn_grid), nn_grid[1].activation), nn_grid, WrappedFunction(x->reshape(x, nn_grid_outputsize, p.p.N_lats, p.p.N_lons, N_batch)))

        nn_sh_inputsize = nn_grid_outputsize + size(input_knowledge_sh_grid, 1) + nn_conv_outputsize

        nn_sh = FlatChain(znormsph, WrappedFunction(x->reshape(x, nn_sh_inputsize, :)), Dense(nn_sh_inputsize, inputsize(nn_sh), nn_sh[1].activation), nn_sh, WrappedFunction(x->SH_zero .* reshape(x, N_levels, p.g.size_SH..., N_batch)), Scale((N_levels,p.g.size_SH...), init_weight=scale_init_weight, init_bias=scale_init_bias))  
    else 
        nn_grid = FlatChain(WrappedFunction(x->reshape(x, nn_grid_inputsize, :)), Dense(nn_grid_inputsize, inputsize(nn_grid), nn_grid[1].activation), nn_grid, WrappedFunction(x->reshape(x, nn_grid_outputsize, p.p.N_lats, p.p.N_lons)))

        nn_sh_inputsize = nn_grid_outputsize + size(input_knowledge_sh_grid, 1) + nn_conv_outputsize

        nn_sh = FlatChain(znormsph, WrappedFunction(x->reshape(x, nn_sh_inputsize, :)), Dense(nn_sh_inputsize, inputsize(nn_sh), nn_sh[1].activation), nn_sh, WrappedFunction(x->SH_zero .* reshape(x, N_levels, p.g.size_SH...)), Scale((N_levels,p.g.size_SH...), init_weight=scale_init_weight, init_bias=scale_init_bias))  
    end
    
    # initialize the transform grid -> SH
    if conv_mode == :unet  # path is different for UNet version, here the transform also gets the UNet output additionally as an input
        transform_sh = TransformSHLayer(QG3.GaussianGridtoSHTransform(p.p, nn_grid_outputsize + conv_out_channels, N_batch=N_batch))
    else # for the other version just the nn_grid output
        transform_sh = TransformSHLayer(QG3.GaussianGridtoSHTransform(p.p, nn_grid_outputsize, N_batch=N_batch))
    end 

    if conv_mode == :unet 

        model = FlatChain(BranchLayer(knowledge_sh_layer, 
                FlatChain(knowledge_grid_layer, znormgrid, BranchLayer(nn_grid, nn_conv), WrappedFunction(x->reduce(vcat, x)), transform_sh)), WrappedFunction(x->reduce(vcat, x)),
                nn_sh)

    elseif conv_mode == :pseudoconv 
        
        model = FlatChain(BranchLayer(knowledge_sh_layer, 
                FlatChain(knowledge_grid_layer, znormgrid, nn_grid, transform_sh),
                nn_conv), 
                WrappedFunction(x->reduce(vcat, x)),
                nn_sh)
    end 

    return PseudoSpectralNet(model)
end

"""
$(TYPEDSIGNATURES)
Initializes a PseudoSpectralNet with the given hyperparameters as keyword arguments
"""
function PseudoSpectralNet(p::QG3Model{T}; N_layers=4, N_Nodes=30, N_channels=9, activation=relu, N_levels=3, conv_mode=:unet, unet_kwargs=NamedTuple(), conv_out_channels=3, N_batch::Int=0, kwargs...) where T

    @assert conv_mode in [:unet, :pseudoconv] "conv_mode needs to be either :unet or :pseudoconv"

    p.g.size_SH

    nn_sh = []
    nn_grid = [] 

    for i ∈ 1:N_layers 
        push!(nn_sh, Dense(N_Nodes, N_Nodes, activation))
        push!(nn_grid, Dense(N_Nodes, N_Nodes, activation))
    end 
    push!(nn_sh, Dense(N_Nodes, N_levels))
    push!(nn_grid, Dense(N_Nodes, 2*N_levels))

    if conv_mode == :unet 
        nn_conv = nothing # we do this later in the other constructor for the UNet case 

    else
        nn_conv = Any[]

        if N_batch > 0 
            error("Batched operation not (yet) supported for PseudoConv layers")
        end 

        mask = QG3.reorder_SH_gpu(QG3.SH_zero_mask(p.p), p.p)
        mask_3d = QG3.reorder_SH_gpu(QG3.SH_zero_mask(p.p, (N_levels,1,1)), p.p)
        size_SH = p.g.size_SH
      
        push!(nn_conv, SHPseudoConv(p.p, (2*N_levels=>N_channels), size_SH, activation))
        for i ∈ 1:N_layers 
            push!(nn_conv, SHPseudoConv(p.p, (N_channels=>N_channels), size_SH, activation))
        end 
        push!(nn_conv, SHPseudoConv(p.p, (N_channels=>conv_out_channels), size_SH, activation))
        nn_conv = FlatChain(FlattenSH(mask_3d), nn_conv..., ExpandSH(mask_3d, (N_levels, size_SH...)))
    end 

    nn_sh = FlatChain(nn_sh...)
    nn_grid = FlatChain(nn_grid...)
    
    PseudoSpectralNet(nn_sh, nn_grid, nn_conv, p; N_levels=N_levels, N_layers=N_layers, N_Nodes=N_Nodes, activation=activation, conv_mode=conv_mode, conv_out_channels=conv_out_channels, unet_kwargs=unet_kwargs, N_batch=N_batch, kwargs...)
end

"""
    PseudoSpectralNet(p::QG3Model, hyperparameters::PSNHyperparameters, N_batch:Int=0, kwargs...)

Initialize a `PseudoSpectralNet` with the `hyperparameters` and additional `kwargs`.
"""
PseudoSpectralNet(p::QG3Model, hyperparameters::PSNHyperparameters, N_batch::Int=0; kwargs...) = PseudoSpectralNet(p; hyperparameters.pars..., N_batch=N_batch, kwargs...)

(m::PseudoSpectralNet)(x, ps, st) = m.model(x, ps, st)

"""
    PSN_RHS(model, qg3p)

The full rhs of the [`NeuralDE`](@ref): dq/dt = -J(q,ψ(q)) + NN((q,ψ(q))). Arguments `model` and `qg3p` the QG3.jl parameters.
"""
struct PSN_RHS{M,P,D} <: Lux.AbstractLuxWrapperLayer{:model}
    model::M
    qg3p::P
    device::D
end 

function (m::PSN_RHS)(x, ps, st)
    ψ = qprimetoψ(m.qg3p, FullDeviceArray(m.device,x))
    nn_res, st = m.model(cat(ψ, x, dims=1), ps, st)
    return - QG3.J(ψ, x, m.qg3p) .+ nn_res, st
end

"""
$(TYPEDSIGNATURES)
Load a PseudoSpectralNet model, if a `SAVE_NAME` is provided a saved/trained parameter vector is loaded from that file. 
"""
function load_psn_ad(qg3p::QG3Model{T}, psn_hyperpars::PSNHyperparameters, DT::T, N_batch=1; SAVE_NAME=nothing, device=NeuralQG3.DetermineDevice(), adjust_for_CPU=false, rng=Random.default_rng(), init=true, S=nothing, kwargs...) where T<:Number

    nn = CUDA.@allowscalar PseudoSpectralNet(qg3p, psn_hyperpars, N_batch; GPU=NeuralQG3.isgpu(device), S=S, kwargs...)
    rhs = NeuralQG3.PSN_RHS(nn, qg3p, device)
    neural_de = ADNeuralDE(model=rhs, alg=NeuralDELux.ADRK4Step(), dt=DT)

    if init
        ps, st = Lux.setup(rng, neural_de)
    else 
        ps = nothing
        st = nothing
    end

    if isnothing(SAVE_NAME)
        ps = NeuralQG3.gpu(device, ComponentArray(ps))
    else 
        if adjust_for_CPU
            ps = load_ps_from_gpu(SAVE_NAME, qg3p.p)
        else 
            ps = load_ps(SAVE_NAME, device)
        end
    end

    return neural_de, ps, st
end 

"""
$(TYPEDSIGNATURES)
Load a PseudoSpectralNet model, if a `SAVE_NAME` is provided a saved/trained parameter vector is loaded from that file.
"""
function load_psn_sciml(qg3p::QG3Model{T}, psn_hyperpars::PSNHyperparameters, DT::T; N_batch::Integer=1, SAVE_NAME=nothing, alg=Tsit5(), process_based::Bool=true, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true), device=NeuralQG3.DetermineDevice(), adjust_for_CPU=false, RELTOL_PREDICT=1e-4, rng=Random.default_rng(), init=true, S=nothing, kwargs...) where T<:Number   

    nn = CUDA.@allowscalar PseudoSpectralNet(qg3p, psn_hyperpars, N_batch; GPU=NeuralQG3.isgpu(device), S, process_based, kwargs...)
    rhs = NeuralQG3.PSN_RHS(nn, qg3p, device)
    neural_de =  SciMLNeuralDE(; model=rhs, alg=alg, sensealg=sensealg, dt=DT, reltol=RELTOL_PREDICT, kwargs...)
    
    if init
        ps, st = Lux.setup(rng, neural_de)
    else 
        ps = nothing
        st = nothing
    end

    if isnothing(SAVE_NAME)
        ps = NeuralQG3.gpu(device, ComponentArray(ps))
    else 
        if adjust_for_CPU
            ps = load_ps_from_gpu(SAVE_NAME, qg3p.p)
        else 
            ps = load_ps(SAVE_NAME, device)
        end
    end

    return neural_de, ps, st
end