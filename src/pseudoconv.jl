using Lux, NNlib, Tullio, QG3

"""
    SHPseudoConv(p::QG3ModelParameters{T}, channels::Pair{Int,Int}, L_max::Integer, size_in, σ=identity; init=glorot_normal) where {T}

Sets up a `SHPseudoConv` layer that performs elementwise multiplication. In spherical harmonics space this corrospodens to convolution, however only when the kernel is zonally symmetric. This layer does not enforce this symmetry constraint, therefore it is "pseudo-convolution". 

# Inputs
* `p`: Parameters of the QG3 Model it's based on 
* `channels`: How many input and output channels the layer should have, i.e. how many pseudoconvolution should be performed in parallel
* `L_max`: Maximum wavenumber of the kernels  
* `size_in`: Size of the input arays in SH domain 
* `σ`: Activation function 
* `init`: initialization function that returns random numbers in the shape of the varargs inputs and type, like e.g. `Flux.glorot_normal``
"""
struct SHPseudoConv{NF,T,P<:Pair{Int,Int},F,S,M} <: Lux.AbstractLuxLayer
    σ::T
    ch::P
    init_func::F
    size_in::S
    SHmask::M
    L_max::Int
end

# implement L_max
function SHPseudoConv(p::QG3ModelParameters{T}, channels::Pair{Int,Int}, size_in, σ=identity; init=glorot_normal, allow_fast_activation::Bool=true) where {T}

    mask_2d = QG3.reorder_SH_gpu(QG3.SH_zero_mask(p, (p.L, p.M)), p)
    σ = allow_fast_activation ? NNlib.fast_act(σ) : σ

    SHPseudoConv{T, typeof(σ), typeof(channels), typeof(init), typeof(size_in), typeof(mask_2d)}(σ, channels, init, size_in, mask_2d, p.L)
end

function Lux.initialparameters(rng::AbstractRNG, c::SHPseudoConv{T}) where T

    init = c.init_func
    size_in = c.size_in 
    L_max = c.L_max
    channels = c.ch 
    mask_2d = Array(c.SHmask) # in case we are on GPU, this is needed on CPU
    
    k = init(T, rng, L_max, (channels[1], channels[2], size_in...)) # why the L_max here 
    b = zeros(T, (channels[2], size_in...))

    k = reshape(k, channels[1], channels[2], :)
    k = k[:,:,reshape(mask_2d, :)]

    b = reshape(b, channels[2], :)
    b = b[:,reshape(mask_2d, :)]

    return (; k, b)
end 

function (m::SHPseudoConv)(x::AbstractArray, ps, st::NamedTuple) 
    @tullio out[ch, ish] := ps.k[ic, ch, ish] * x[ic, ish] 
    return m.σ.(out + ps.b), st
end

function Lux.parameterlength(c::SHPseudoConv) 
    return prod((c.ch[1], c.ch[2], c.size_in...)) + prod((c.ch[2], c.size_in...)) # this is wrong! 
end

Lux.statelength(c::SHPseudoConv) = 0

function Base.show(io::IO, l::SHPseudoConv)
    print(io, "SHPseudoConv(", l.ch, ")")
end

inputsize(m::SHPseudoConv) = size(m.k)
outputsize(m::SHPseudoConv) = size(m.k)

function default_shconvnet(p::QG3Model, channels::Pair{Int,Int}=(6=>3); activation=swish) 
    mask_3d = QG3.reorder_SH_gpu(QG3.SH_zero_mask(p.p, (channels[2],1,1)), p.p)
    size_SH = p.g.size_SH
    return Chain(FlattenSH(mask_3d), SHPseudoConv(p.p, (channels[1]=>9), size_SH, activation), SHPseudoConv(p.p, (9=>6), size_SH, activation), SHPseudoConv(p.p, (6=>channels[2]), size_SH, identity), ExpandSH(mask_3d))
end

