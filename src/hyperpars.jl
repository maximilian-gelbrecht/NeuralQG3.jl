
"""
    PSNHyperparameters(pars::NamedTuple)

Wraps around a `NamedTuple` to save the hyperparmeters of a PseudoSpectralNet. 
"""
struct PSNHyperparameters
    pars::NamedTuple 
end 

"""
    parse_pars(pars::NamedTuple; kwargs...)

Processes an input `pars` to seperate architecture specific hyperparameters into an instance of `PSNHyperparameters` and training/optimziation specific parameters into a `NamedTuple`. Additional keyword arguments are added to the `PSNHyperparameters`.
"""
function parse_pars(pars; kwargs...)
    
    pars_keys = keys(pars)

    psn_args = NamedTuple()
    other_hyperpars = NamedTuple()
    for par_key in pars_keys
        if par_key == :kernel 
            psn_args = (psn_args..., unet_kernel=pars[par_key],)
        elseif par_key == :N_channels
            psn_args = (psn_args..., unet_channels=pars[par_key],)
        elseif par_key == :N_Nodes 
            psn_args = (psn_args..., N_Nodes=pars[par_key],)
        elseif par_key == :N_layers 
            psn_args = (psn_args..., N_layers=pars[par_key],)
        elseif par_key == :activation 
            func = pars[:activation ]
            if func == "relu"
                func = NNlib.relu
            elseif func == "selu"
                func = NNlib.selu 
            elseif func == "swish"
                func = NNlib.swish
            elseif func == "tanh"
                func = NNlib.tanh
            else
                error("unkonwn activation function")
            end
            psn_args = (psn_args..., activation=func,)
        elseif par_key == :DT_FAC
            other_hyperpars = (other_hyperpars..., DT_FAC=pars[par_key],)
        elseif par_key == :learning_rate
            other_hyperpars = (other_hyperpars..., η=pars[par_key],)
        elseif par_key == :N_batch 
            other_hyperpars = (other_hyperpars..., N_batch=pars[par_key],)
        elseif par_key == :scale
            other_hyperpars = (other_hyperpars..., SCALE=pars[par_key],)
        elseif par_key == :znorm 
            psn_args = (psn_args..., znorm=pars[par_key],)
        elseif par_key == :weight_decay
            other_hyperpars = (other_hyperpars..., weight_decay=pars[par_key],) 
        elseif par_key == :gamma 
            other_hyperpars = (other_hyperpars..., γ=pars[par_key],)           
        elseif par_key == :tau
            other_hyperpars = (other_hyperpars..., τ_max=pars[par_key],)    
        elseif par_key == :data_length
            other_hyperpars = (other_hyperpars..., data_length=pars[par_key])  
        elseif par_key == :N_epoch_sciml
            other_hyperpars = (other_hyperpars..., N_epoch_sciml=pars[par_key])
        elseif par_key == :N_epoch_ad
            other_hyperpars = (other_hyperpars..., N_epoch_ad=pars[par_key])
        else
            error("Unknown Hyperpar specified!")
        end
    end

    return PSNHyperparameters(merge(psn_args, kwargs)), training_hyperpars(;other_hyperpars...)
end 

function PSNHyperparameters(; N_layers=3, N_Nodes=30, N_channels=3, unet_channels=[24,48,96,96], unet_kernel=(3,3), unet_convblock=ConvBlock, activation=NNlib.swish, additional_knowledge=false, conv_mode=:unet, znorm=true, kwargs...)
    PSNHyperparameters((N_layers=N_layers, N_Nodes=N_Nodes, N_channels=N_channels, unet_kwargs=(kernel=unet_kernel, N_channels=unet_channels, downconv=unet_convblock, upconv=unet_convblock), activation=activation, additional_knowledge=additional_knowledge, conv_mode=conv_mode, znorm=znorm, kwargs...))
end 

"""
$(TYPEDSIGNATURES)
Returns a NamedTuple with all optimization/training specific hyperparameters.
"""
function training_hyperpars(; DT_FAC=10, η=1f-3, N_batch=8, SCALE=false, γ = 1f-6, τ_max=2, N_epoch_ad=20, N_epoch_sciml=1, N_epoch_offset=0, data_length=200f0)
    return (DT_FAC=DT_FAC, η=η, N_batch=N_batch, SCALE=SCALE, γ=γ, τ_max=τ_max, N_epoch_ad=N_epoch_ad, N_epoch_sciml=N_epoch_sciml, N_epoch_offset=N_epoch_offset, data_length=data_length)
end 

"""
$(TYPEDSIGNATURES)
Returns all file names used during the I/O of the model. 
"""
function setup_savenames(arg_string=[]; default_name::Union{String,Nothing}="psn-model", save_dir::String="", job_id::Union{Integer,Nothing}=nothing, verbose::Bool=true)

    SAVE_NAME = [save_dir]
    if length(arg_string) >= 1
        push!(SAVE_NAME,arg_string[1])
        if !isnothing(job_id) 
            push!(SAVE_NAME,"-$job_id")
        end 
    else
        @assert !isnothing(default_name) "A name must be specificed!" 
        push!(SAVE_NAME,default_name)
    end 

    SAVE_NAME_SOL = string(SAVE_NAME...,"-sol.jld2")
    SAVE_NAME_MODEL = string(SAVE_NAME..., "-model-struct.jld2")
    SAVE_NAME_RESULTS = string(SAVE_NAME..., "-training-results.jld2")
    SAVE_NAME_RESULTS_2 = string(SAVE_NAME..., "-training-results-2.jld2")

    SAVE_NAME = string(SAVE_NAME...,"-model-pars.jld2")
    
    if verbose 
        println(string("using data / saving data as ",SAVE_NAME))
    end

    return (SAVE_NAME=SAVE_NAME, SAVE_NAME_MODEL=SAVE_NAME_MODEL, SAVE_NAME_RESULTS=SAVE_NAME_RESULTS, SAVE_NAME_RESULTS_2=SAVE_NAME_RESULTS_2, SAVE_NAME_SOL=SAVE_NAME_SOL)
end 
