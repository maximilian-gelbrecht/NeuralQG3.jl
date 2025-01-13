using StatsBase, Lux

const NAME_TYPE = Union{Nothing, String, Symbol}

"""
Old version of Lux' Chain that flattens the layers. This is used as legacy code becaused we started the project with 
Chains constructed like this and saved parameters like that. The new Chain breaks previously saved models. 

This will be removed in future version, but is used to archive the results of the paper
"""
struct FlatChain{L<:NamedTuple,S} <: AbstractLuxWrapperLayer{:layers}
    layers::L 
    name::S
end

function FlatChain(xs...; name::NAME_TYPE=nothing, disable_optimizations::Bool=false)
    xs = disable_optimizations ? xs : flatten_lux_chain(xs)
    length(xs) == 0 && return NoOpLayer()
    length(xs) == 1 && return first(xs)
    return FlatChain(Lux.Utils.named_tuple_layers(xs...), name)
end

FlatChain(xs::AbstractVector; kwargs...) = FlatChain(xs...; kwargs...)

function FlatChain(nt::NamedTuple; disable_optimizations::Bool=true, name::NAME_TYPE=nothing)
    if !disable_optimizations
        throw(ArgumentError("FlatChain(::NamedTuple) is not compatible with disable_optimizations=true"))
    end
    return FlatChain(nt, name)
end

function FlatChain(; disable_optimizations::Bool=true, name::NAME_TYPE=nothing, kwargs...)
    return FlatChain((; kwargs...); disable_optimizations, name)
end

function flatten_lux_chain(layers::Union{AbstractVector, Tuple})
    new_layers = []
    for l in layers
        f = flatten_lux_chain(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
        elseif f isa Function
            if !hasmethod(f, (Any, Any, NamedTuple))
                f === identity && continue
                push!(new_layers, WrappedFunction{:direct_call}(f))
            else
                push!(new_layers, WrappedFunction{:layer}(f))
            end
        elseif (f isa Chain) || (f isa FlatChain)
            append!(new_layers, f.layers)
        elseif f isa NoOpLayer
            continue
        else
            push!(new_layers, f)
        end
    end
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

flatten_lux_chain(x) = x

(c::FlatChain)(x, ps, st::NamedTuple) = Lux.applychain(c.layers, x, ps, st)

Base.keys(c::FlatChain) = Base.keys(getfield(c, :layers))

Base.getindex(c::FlatChain, i::Int) = c.layers[i]
Base.getindex(c::FlatChain, i::AbstractArray) = FlatChain(Utils.index_namedtuple(c.layers, i))

function Base.getproperty(c::FlatChain, name::Symbol)
    hasfield(typeof(c), name) && return getfield(c, name)
    layers = getfield(c, :layers)
    hasfield(typeof(layers), name) && return getfield(layers, name)
    throw(ArgumentError("$(typeof(c)) has no field or layer $name"))
end

Base.length(c::FlatChain) = length(c.layers)
Base.lastindex(c::FlatChain) = lastindex(c.layers)
Base.firstindex(c::FlatChain) = firstindex(c.layers)

outputsize(c::FlatChain) = outputsize(c.layers[end])

"""
    FlattenSH{S}(mask) <: Lux.AbstractLuxLayer   

Constructs a non-trainable layer that flattens an 3d SH coefficent array (lvl x l x m) into a 2d one (lvl x lm). The input mask is a boolean mask of the valid SH coefficient entries in the 3d array. The input `mask` is either a 3D mask that will converted into a 2D mask, or a 2D mask. 
"""
struct FlattenSH{S} <: Lux.AbstractLuxLayer
    mask::S

    function FlattenSH(mask::S) where S
        if ndims(mask)==3 
            mask_2d = mask[1,:,:]
            for i in axes(mask,1)
                if mask[i,:,:] != mask_2d
                    @warn "Mask not equal over all levels, this will be ignored and only mask[1,:,:] will be used"
                end 
            end
            return new{typeof(mask_2d)}(mask_2d)
        elseif ndims(mask)==2
            return new{S}(mask)
        else 
            error("Mask needs to be 2d or 3d")
        end
    end     
end 

Lux.initialparameters(rng::AbstractRNG, f::FlattenSH) = NamedTuple()
Lux.parameterlength(f::FlattenSH) = 0 

(m::FlattenSH)(x, ps, st) = flatten_SH(x, m.mask), st

# there is a difference between indexing of Arrays and CuArrays
flatten_SH(A, mask) = view(reshape(A, size(A,1), :),:,reshape(mask,:))
flatten_SH(A::CuArray, mask) = view(A,:,mask)

"""
    ExpandSH(mask) <: Lux.AbstractLuxLayer

Construcs a non-trainable layer that reshaps an array of SH coefficents that was flattened into a 1D array (e.g. by [`FlattenSH`](@ref)) into a 2D array
"""
struct ExpandSH{S,T} <: Lux.AbstractLuxLayer
    mask::S
    size::T
end 

Lux.initialparameters(rng::AbstractRNG, e::ExpandSH) = NamedTuple()
Lux.parameterlength(e::ExpandSH) = 0 

outputsize(m::ExpandSH) = size(m.mask)

ExpandSH(mask::S) where S = ExpandSH(mask, size(mask))

(m::ExpandSH)(x, ps, st) = expand_flattened_SH(x, m.mask, m.size), st

function expand_flattened_SH(A, mask, size) 
    B = zeros(eltype(A), size)
    B[mask] = A 
    return B
end

function expand_flattened_SH(A::CuArray, mask, size) 
    B = CUDA.zeros(eltype(A), size)
    B[mask] = A 
    return B
end

Zygote.@adjoint function expand_flattened_SH(A, mask, size)
    return (expand_flattened_SH(A, mask, size), Δ->(flatten_SH(Δ,reshape(mask[1,:,:],:)),nothing,nothing))
end

# masks are different for each of the functions and also between CPU and GPU
Zygote.@adjoint function expand_flattened_SH(A::CuArray, mask, size)
    return (expand_flattened_SH(A, mask, size), Δ->(flatten_SH(Δ,view(mask,1,:,:)),nothing,nothing))
end

glorot_normal(T, rng::AbstractRNG, L_max, dims...) = randn(rng, T, dims...) .* T(sqrt(24.0f0 / ((L_max+0.5)*L_max))) # 0.5 * (2L_max + 1) * L_max
glorot_normal(T, L_max, dims...) = glorot_normal(T, rng_from_array(), L_max, dims...)

"""
    rng_from_array([x])
Create an instance of the RNG most appropriate for `x`.
The current defaults are:
- `x isa AbstractArray`
  - Julia version is < 1.7: `Random.GLOBAL_RNG`
  - Julia version is >= 1.7: `Random.default_rng()`
- `x isa CuArray`: `CUDA.default_rng()`
When `x` is unspecified, it is assumed to be a `AbstractArray`.
"""
rng_from_array(::AbstractArray) = rng_from_array()
rng_from_array(::CuArray) = CUDA.default_rng()
if VERSION >= v"1.7"
  rng_from_array() = Random.default_rng()
else
  rng_from_array() = Random.GLOBAL_RNG
end


"""
    SPHZNormalize(p::QG3.QG3ModelParameters)

Returns a non-trainable layer that performs a Z Normalization of the 2D SPH coefficient arrays, therefore it includes a bias correction. 
"""
struct SPHZNormalize{T,S} <: Lux.AbstractLuxLayer
    N_nonzero::T
    N_zero::T
    mask::S
end 

function SPHZNormalize(p::QG3.QG3ModelParameters; N_batch::Integer=0)

    if N_batch > 0
        mask = QG3.reorder_SH_gpu(QG3.SH_zero_mask(p, (1,1,1, N_batch); N_batch=N_batch),  p)
        N_nonzero = sum(mask[:,:,:,1])
        N_zero = prod(size(mask[:,:,:,1])) - N_nonzero
    else    
        mask = QG3.reorder_SH_gpu(QG3.SH_zero_mask(p, (1,1,1)),  p)
        N_nonzero = sum(mask)
        N_zero = prod(size(mask)) - N_nonzero
    end 

    SPHZNormalize(N_nonzero, N_zero, mask)
end 

function (sph::SPHZNormalize)(x, ps, st::NamedTuple)
    μ, σ = sph_z_normaliziation(x, sph.N_nonzero, sph.N_zero)

    return ((x .- μ)./σ) .* sph.mask, st
end 

"""
    sph_z_normaliziation(data::AbstractArray{T,3}, N_nonzero::Integer, N_zero::Integer)

When computing mean and σ of SPH arrays with many zeros, the results are biased by the many zeroes in the array. 
"""
function sph_z_normaliziation(data::AbstractArray{T,NA}, N_nonzero::Integer, N_zero::Integer) where {T,NA}
    N = N_nonzero + N_zero 

    μ = mean(data,dims=(2,3))
    μ_corrected = μ .* (N/N_nonzero)

    σ = std(data,dims=(2,3), mean=μ_corrected)
    σ_corrected = sqrt.( (N/N_nonzero)*σ.^2 + (N_zero/N_nonzero).*(μ_corrected.^2))

    T.(μ_corrected), T.(σ_corrected)
end 

"""
    GridZNormalize()

Returns a non-trainable layer that performs a z-normalization in each of the horizontal layers seperately. The vertical dimension is assumed to be the first one. 
"""
struct GridZNormalize <: Lux.AbstractLuxLayer
end 

function (sph::GridZNormalize)(x, ps, st::NamedTuple)
    μ = StatsBase.mean(x, dims=(2,3))
    σ = StatsBase.std(x, dims=(2,3), mean=μ)
    return (x .- μ)./σ, st
end 

"""
    PermutedimLayer(order)

Permutes the dimensions of the passed array to `order`.

## Arguments

  - `order`: The new order, as in `Base.permutedims`

## Inputs

  - `x`: AbstractArray of any shape which can be permuted with `order`

## Returns

  - AbstractArray with dimensions permuted
  - Empty `NamedTuple()`
"""
struct PermutedimLayer{N} <: Lux.AbstractLuxLayer
    order::NTuple{N, Int}
end

@inline function (r::PermutedimLayer)(x::AbstractArray, ps, st::NamedTuple)
    return permutedims(x, r.order), st
end

function Base.show(io::IO, r::PermutedimLayer)
    return print(io, "PermutedimLayer(order = (", join(r.order, ")"))
end

"""
    ScaledInitDense{D,I} <: Lux.AbstractLuxLayer

A regular dense layer but with the weights initialized reduced by `init_scale`
"""
struct ScaledInitDense{D,I} <: Lux.AbstractLuxLayer
    dense::D 
    init_scale::I
end 

(m::ScaledInitDense)(x,ps,st) = m.dense(x,ps,st)

function Lux.initialparameters(rng::AbstractRNG, m::ScaledInitDense) 
    ps = Lux.initialparameters(rng, m.dense)
    ps[:weight] .*= m.init_scale
    return ps 
end 

Base.show(io::IO, d::ScaledInitDense) = Base.show(io, d.dense)
outputsize(m::ScaledInitDense, x::AbstractArray) = outputsize(m.dense, x)

"""
    GenerateScaleInitializer{T}(scale::T)

Helper struct to initialize a `Scale` layer from Lux with a certain array input `scale`. The struct is overloaded to be the initializer `(rng, dims)->scale`
"""
struct GenerateScaleInitializer{T}
    scale::T
end 

GenerateScaleInitializer(scale::CuArray) = GenerateScaleInitializer(cpu(scale)) # for Lux the generator needs to return a CPU array at first, it is transferred to GPU later 

function (m::GenerateScaleInitializer)(rng, dims...)
    @assert ((size(m.scale) == dims) || (size(m.scale) == dims[1:end-1])) "Initial scale has wrong size"
    return m.scale 
end

