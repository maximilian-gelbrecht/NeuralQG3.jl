using EllipsisNotation, StatsBase, ComponentArrays
using NeuralDELux, NNlib

inputsize(m::FlatChain) = inputsize(m.layers[1])
inputsize(m::Lux.Dense) = m.in_dims

outputsize(m::Lux.Dense) = m.out_dims

repeat_batch(x::T, N_batch::Int) where T =  N_batch > 0 ? repeat(x,1,1,1,N_batch) : x 

"""
    slice_and_batch_trajectory(t::AbstractVector, x, N_batch::Integer)

Slice a single trajectory into multiple ones for the batched dataloader.
"""
function slice_and_batch_trajectory(t::AbstractVector, x, N_batch::Integer)

    @assert N_batch > 0
    
    N_t = length(t)
    N_t_batch = div(N_t, N_batch)

    trajs = []
    for i=1:N_batch
        push!(trajs, (t[1+(i-1)*N_t_batch:i*N_t_batch], x[..,1+(i-1)*N_t_batch:i*N_t_batch]))
    end

    return trajs
end 

"""
    GridForecast(data, transform; model=nothing, threshold::Number=1e-2, N_forecast::Integer=50, N_avg=20, metric::String="norm", modes::Union{Tuple,String}=("forecast_delta",), trajectory_call::Bool=false, input_SH::Bool=true, data_SH::Bool=true, output_SH::Bool=true)
    
Sets up an instance of `GridForecast` to make evaluate predictions of `model` on `data` on the grid with the spherical harmonics to grid `transform`. Once inialized call the `GridForecast` with: `gf(model, ps, st)`. Currenlty supporeted metrics are: `forecast_delta`, `forecast_length`, `average_forecast_delta`, `average_forecast_length` and `latitude_delta`. Additional keywords govern if the input/output of the model is in the spherical harmonics or grid domain. Appropiate transforms are then chosen automatically. 
"""
@kwdef struct GridForecast{D,T,MO,TG,S,M}
    data::D
    transform::TG=nothing
    model::MO=nothing
    threshold::T=1e-2
    N_forecast::Integer=50
    N_avg::Integer=20
    metric::S="norm"
    modes::M=("forecast_delta",)
    trajectory_call::Bool=false
    input_SH::Bool=true
    output_SH::Bool=true
    data_SH::Bool=true
end

function GridForecast(data, transform; model=nothing, threshold::Number=1e-2, N_forecast::Integer=50, N_avg=20, metric::String="norm", modes::Union{Tuple,String}=("forecast_delta",), trajectory_call::Bool=false, input_SH::Bool=true, data_SH::Bool=true, output_SH::Bool=true)
    
    if typeof(modes) <: String 
        modes = (modes, )
    end

    GridForecast(data, transform, model, threshold, N_forecast, N_avg, metric, modes, trajectory_call, input_SH, output_SH, data_SH)
end

function (gf::GridForecast)(model, ps, st)

    t, x = gf.data 
    N_forecast = gf.N_forecast
    modes = gf.modes 
    
    if !isnothing(gf.model)
        forecast_model = gf.model
    else 
        forecast_model = model 
    end 

    @assert length(t) >= N_forecast 

    results = NamedTuple()

    if gf.trajectory_call
        predict_call = (x, ps, st) -> NeuralDELux.trajectory(forecast_model, x, ps, st)
    else 
        predict_call = forecast_model 
    end 

    if gf.output_SH
        predict = (x, ps, st) -> (QG3.transform_grid_data(predict_call(x, ps, st)[1], gf.transform), nothing) # the nothign is there to emulate a second return arugment as the lux models have
    else 
        predict = predict_call
    end

    if ("forecast_delta" in modes) || ("forecast_length" in modes)

        forecast_delta = forecast_δ(predict(input_data(gf, 1), ps, st)[1], ground_truth_data(gf,1)[2], gf.metric)[:]

        if "forecast_delta" in modes
            results = (forecast_delta=Array(forecast_delta), results...,)
        end 
        
        if "forecast_length" in modes 
            results = (forecast_length=findfirst(forecast_delta .> gf.threshold), results...)
        end 
    end 

    if ("latitude_delta" in modes)
        forecast_delta = forecast_latitude_δ(predict(input_data(gf, 1), ps, st)[1], ground_truth_data(gf,1)[2], gf.metric)
        results = (latitude_delta=Array(forecast_delta), results...,)
    end

    if ("average_forecast_length" in modes) || ("average_forecast_delta" in modes)

        @assert length(t) >= N_forecast + gf.N_avg 

        avg_forecast = 0.0

        if ("average_forecast_delta" in modes)
            delta = zeros(eltype(x), N_forecast)
        end 

        for i=1:gf.N_avg 
            forecast_delta = forecast_δ(predict(input_data(gf, i), ps, st)[1], ground_truth_data(gf,i)[2], gf.metric)[:]
            
            if ("average_forecast_delta" in modes)
                delta += Array(forecast_delta)
            end

            findex = findfirst(forecast_delta .> gf.threshold)
            if isnothing(findex) # incase no forecast_delta is larger, we have a very good forecast
                avg_forecast += N_forecast + 1
            else 
                avg_forecast += findfirst(forecast_delta .> gf.threshold)
            end
        end 
        avg_forecast /= gf.N_avg 

        if ("average_forecast_delta" in modes)
            delta ./= gf.N_avg 
            results = (average_forecast_delta=delta, results...,)
        end 

        if ("average_forecast_length" in modes) 
            avg_forecast /= gf.N_avg 
            results = (average_forecast_length=avg_forecast, results...,)
        end
    end 

    return results 
end 

function input_data(gf::GridForecast, i::Integer)

    (t, x) = gf.data
    N_forecast = gf.N_forecast

    @assert i <= length(t) - N_forecast

    if gf.input_SH    
        if gf.data_SH 
            return (t[..,i:N_forecast+i-1], x[..,i:N_forecast+i-1])
        else 
            error("Input data for model is SH, but no SH data provided")
        end 
    else    
        if gf.data_SH
            return (t[..,i:N_forecast+i-1], QG3.transform_grid_data(x[..,i:N_forecast], gf.transform))
        else 
            return (t[..,i:N_forecast+i-1], x[..,i:N_forecast+i-1])
        end 
    end 
end 

function ground_truth_data(gf::GridForecast, i::Integer)

    (t, x) = gf.data
    N_forecast = gf.N_forecast

    @assert i <= length(t) - N_forecast

    if gf.data_SH 
        return (t[..,i:N_forecast+i-1], QG3.transform_grid_data(x[..,i:N_forecast+i-1], gf.transform))
    else 
        return (t[..,i:N_forecast+i-1], x[..,i:N_forecast+i-1])
    end
end 

"""
    forecast_δ(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String="both") where {T,N}

Assumes that the last dimension of the input arrays is the time dimension and `N_t` long. Returns an `N_t` long array, judging how accurate the prediction is. 

Supported modes: 
* `"mean"`: mean between the arrays
* `"maximum"`: maximum norm 
* `"norm"`: normalized, similar to the metric used in Pathak et al 
"""
function forecast_δ(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String="norm"; dims=1:N-1) where {T,N}

    if !(size(prediction) == size(truth))  # if prediction is to short insert Inf, this happens espacially when the solution diverges, so Inf also has a physical meaning here 
        prediction_temp = Inf .* typeof(prediction)(ones(eltype(prediction), size(truth)))
        prediction_temp[..,1:size(prediction,ndims(prediction))] = prediction 
        prediction = prediction_temp 
    end 

    if !(mode in ["mean","largest","both","norm"])
        error("mode has to be either 'mean', 'largest' or 'both', 'norm'.")
    end

    δ = abs.(prediction .- truth)

    if mode == "mean"
        return mean(δ, dims=dims)
    elseif mode == "maximum"
        return maximum(δ, dims=dims)
    elseif mode == "norm"
        return sqrt.(sum((prediction .- truth).^2, dims=dims))./sqrt(mean(sum(abs2, truth, dims=dims)))
    else
        return (mean(δ, dims=dims), maximum(δ, dims=1))
    end
end

function get_layer_i(x, i::Integer)
    x[layer_name(i)]
end 

layer_name(i::Integer) = Symbol(string("layer_",i))

"""
    forecast_latitude_δ(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String="both") where {T,N}

Assumes that the last dimension of the input arrays is the time dimension and `N_t` long. Returns an `N_t` long array, judging how accurate the prediction by latitude is. 

Supported modes: 
* `"mean"`: mean between the arrays
* `"maximum"`: maximum norm 
* `"norm"`: normalized, similar to the metric used in Pathak et al 
"""
function forecast_latitude_δ(prediction::AbstractArray, truth::AbstractArray, mode::String="norm")
    return forecast_δ(prediction, truth, mode, dims=(1,3))
end

"""
    load_ps(SAVE_NAME, dev::AbstractDevice)

Load the parameters from `SAVE_NAME`, if `dev == DeviceCUDA()`, they are directly transfered to a GPU.
"""
function load_ps(SAVE_NAME, dev::DeviceCUDA)
    @load SAVE_NAME ps_save 
    return cu(ComponentArray(ps_save))
end

function load_ps(SAVE_NAME, dev::DeviceCPU)
    @load SAVE_NAME ps_save 
    return ComponentArray(ps_save)
end 


"""
    load_ps_from_gpu(SAVE_NAME)

Load the parameters from `SAVE_NAME`. With the current model there is a differnece between the parameter vector of CPU and GPU models in exactly one layer. This is due to the fact how the spurious elements of the SH coefficient matrices are handled. This routine loads a parameter vector that was trained on GPU, and converts it so that it can be used on a CPU model. 
"""
function load_ps_from_gpu(SAVE_NAME, p::QG3ModelParameters)
    ps = load_ps(SAVE_NAME, DeviceCPU())
    return adjust_ps_for_cpu(ps, p)
end 

function adjust_ps_for_cpu(ps, p::QG3ModelParameters, layer_name=:layer_11)
    ps_scale = ps[layer_name] 
    new_scale_ps = (weight=QG3.reorder_SH_cpu(ps_scale[:weight],p), bias=QG3.reorder_SH_cpu(ps_scale[:bias],p),)

    ps = NamedTuple(ps)
    ps = delete(ps, layer_name)
    ps = (;ps..., layer_name => new_scale_ps)

    return ComponentArray(ps)
end 

# taken from PR#27725 of Julia.Base that wasn't merged yet
"""
    delete(a::NamedTuple, field::Symbol)
Construct a new named tuple from `a` by removing the named field.
```jldoctest
julia> delete((a=1, b=2, c=3), :a)
(b = 2, c = 3)
julia> delete((a=1, b=2, c=3), :b)
(a = 1, c = 3)
```
"""
function delete(a::NamedTuple{an}, field::Symbol) where {an}
    names = Base.diff_names(an, (field,))
    NamedTuple{names}(a)
end

"""
$(TYPEDSIGNATURES)
Takes a trajectory (t,x) and adds batch dimension `N_batch` times
"""
function trajectory_insert_batchdim(trajectory; N_batch=1)
    (t, x) = trajectory

    x = insert_dim(x, 4)
    return (t,repeat(x, 1,1,1,N_batch,1))
end 

function insert_dim(A, i_dim)
    s = [size(A)...]
    insert!(s, i_dim, 1)
    return reshape(A, s...)
end

# this is an adjustment due to an breaking update in Lux, and the old parameters still stored with Nx1 matrices as the bias
import Lux.LuxLib.fused_dense_bias_activation
import Lux.LuxLib.fused_conv_bias_activation
import Lux.LuxLib.bias_activation!!

fused_dense_bias_activation(σ::F, weight::AbstractMatrix, x::AbstractMatrix,
        b::AbstractMatrix) where {F} = fused_dense_bias_activation(σ, weight, x, b[:])

fused_conv_bias_activation(σ::F, weight::AbstractArray{wT, N}, x::AbstractArray{xT, N},
            b::AbstractArray, cdims::NNlib.ConvDims) where {F, N, wT, xT} = fused_conv_bias_activation(σ, weight, x, b[:], cdims)

bias_activation!!(σ::F, x::AbstractArray, bias::AbstractArray) where F = bias_activation!!(σ, x, bias[:])