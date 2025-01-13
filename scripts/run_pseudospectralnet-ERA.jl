using Pkg
Pkg.activate("scripts/")

using QG3, NeuralQG3, JLD2, Dates, NNlib, LuxCUDA, NeuralDELux, ComponentArrays, SlurmHyperopt, BenchmarkTools, OrdinaryDiffEq, ParameterSchedulers, Optimisers, Random, SciMLSensitivity, Lux, Zygote, Statistics, CUDA, NODEData, Printf, EllipsisNotation

const GPU = CUDA.functional()
const DEV = NeuralQG3.DetermineDevice(gpu=GPU)

if length(ARGS) > 1
    println("Loading Hyperparameters...")
    @load "hyperopt_psn.jld2" sho 
    i_job = parse(Int,ARGS[2])
    pars = sho[i_job]  

    psn_hyperpars, other_hyperpars = parse_pars(pars; additional_knowledge=true)

    TRAIN = true
    CONTINUE_TRAINING = false
    SETUP_SCIML = true
    TRAIN_SCIML = true
    PRETRAIN_NAME = ""
else 
    psn_hyperpars = PSNHyperparameters(additional_knowledge=true, unet_convblock=NeuralQG3.LongLatConvBlock)
    other_hyperpars = NeuralQG3.training_hyperpars(N_epoch_ad=200, N_epoch_sciml=10, τ_max=6, N_batch=14)

    TRAIN = true
    CONTINUE_TRAINING = false
    SETUP_SCIML = true
    TRAIN_SCIML = true
    i_job = nothing
    PRETRAIN_NAME = "psn-era"
end
(; DT_FAC, η, N_batch, SCALE, γ, τ_max, N_epoch_ad, N_epoch_sciml) = other_hyperpars

ERA5_FILE = "/p/projects/ou/labs/ai/reanalysis/era5/T42/streamfunction/hourly/ERA5-sf-200500800hPa-90.nc"

Random.seed!(123456)
RELTOL_PREDICT = 1e-3

(; SAVE_NAME, SAVE_NAME_MODEL, SAVE_NAME_RESULTS, SAVE_NAME_SOL) = NeuralQG3.setup_savenames(ARGS; job_id=i_job)
pre_save_names = NeuralQG3.setup_savenames([]; default_name=PRETRAIN_NAME, job_id=i_job)
PRE_SAVE_NAME = pre_save_names[:SAVE_NAME]
PRE_SAVE_NAME_MODEL = pre_save_names[:SAVE_NAME_MODEL]

if CONTINUE_TRAINING 
    println("Loading Hyperpar Config from save file...")

    if isnothing(PRETRAIN_NAME)
        @load SAVE_NAME_MODEL psn_hyperpars
    else 
        @load PRE_SAVE_NAME_MODEL psn_hyperpars
    end 
else 
    @save SAVE_NAME_MODEL psn_hyperpars
end

# load QG3 model components 
S, qg3ppars, ψ_0, q_0 = load_data("T42", GPU=GPU)
__ = nothing 

# pre-computations are partially performed on CPU, so we have to allow scalarindexing
qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=N_batch)
T = eltype(qg3p)

(t, q) = NeuralQG3.compute_ERA_data(ERA5_FILE, qg3p)
q = GPU ? T.(QG3.reorder_SH_gpu(q, qg3ppars)) : T.(q)
DT = T(t[2] - t[1])

println("Using DT = $DT")

train, valid, test = CUDA.@allowscalar NODEData.SingleTrajectoryBatchedOSADataloader(q, t, N_batch, valid_set=0.1, test_set=0.1)

nn = CUDA.@allowscalar PseudoSpectralNet(qg3p, psn_hyperpars, N_batch; GPU=GPU, S=S)
rhs = NeuralQG3.PSN_RHS(nn, qg3p, NeuralQG3.DetermineDevice(gpu=GPU))

neural_de = ADNeuralDE(model=rhs, alg=NeuralDELux.ADRK4Step(), dt=DT)

rng = Random.default_rng()
ps, st = Lux.setup(rng, neural_de)
ps = NeuralQG3.gpu(DEV, ComponentArray(ps))

if CONTINUE_TRAINING
    if isnothing(PRETRAIN_NAME)
        ps = NeuralQG3.load_ps(SAVE_NAME, DEV)
    else 
        ps = NeuralQG3.load_ps(PRE_SAVE_NAME, DEV)
    end 
end 

loss = NeuralDELux.least_square_loss_ad
loss_val = loss(train[1], neural_de, ps, st)

opt = Optimisers.AdamW(η, (9f-1, 9.99f-1), γ)
opt_state = Optimisers.setup(opt, ps)

η_schedule = try
    SinExp(λ0=η,λ1=1f-5,period=30,γ=0.985f0) # old syntax
catch e 
    SinExp(l0=η,l1=1f-5,period=40,decay=0.995f0) # new syntax
end

valid_trajectory = NODEData.get_trajectory(valid, 100; N_batch=N_batch)

if TRAIN 
    loss_val, gs = @time Zygote.withgradient(ps -> loss(train[1], neural_de, ps, st)[1], ps)
end

if TRAIN
    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, trajectory_call=true)

    neural_de, ps, st, results_ad = NeuralDELux.train!(neural_de, ps, st, loss, train, opt_state, η_schedule; τ_range=2:2, N_epochs=N_epoch_ad, shuffle_data_order=false, verbose=true, additional_metric=grid_forecast, valid_data=valid, save_name=SAVE_NAME)

    println("... finished AD training")
    @save SAVE_NAME_RESULTS results_ad
end

if SETUP_SCIML 
    # further epochs with Tsit single (it's more accurate)
    N_batch_sciml = N_batch
    qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=N_batch_sciml)
    
    neural_de_sciml, __, __ = NeuralQG3.load_psn_sciml(qg3p, psn_hyperpars, DT; N_batch=N_batch_sciml, device=DEV, init=false, S=S, alg=NeuralDELux.MultiStepRK4(), RELTOL_PREDICT=RELTOL_PREDICT)

    loss_sciml = CUDA.@allowscalar NeuralQG3.SpectralFilteringLoss(qg3p, 2:τ_max; N_batch=N_batch_sciml)
 
    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, N_forecast=175, input_SH=true, output_SH=true, data_SH=true, trajectory_call=false)

    if TRAIN_SCIML
        # precompile
        begin  
            #valid_3 = remake_dataloader(valid, 7) 
            #valid_3 = valid_3[1]
            valid_3 = valid[1]
            #loss_val, gs = CUDA.@time Zygote.withgradient(ps -> loss_sciml(valid_3, neural_de_sciml, ps, st)[1], ps);
            loss_val, gs = CUDA.@time Zygote.withgradient(ps -> loss_sciml[1](valid_3, neural_de_sciml, ps, st)[1], ps);
        end 

        neural_de_sciml, ps, st, results_sciml = NeuralDELux.train!(neural_de_sciml, ps, st, loss_sciml, train, opt_state, η_schedule; τ_range=2:τ_max, N_epochs=N_epoch_sciml, verbose=true, additional_metric=grid_forecast, valid_data=valid, scheduler_offset=N_epoch_ad, save_name=SAVE_NAME)

        # eval the forecast length
        println("...finished SciML training...")
        @save SAVE_NAME_RESULTS results_ad results_sciml
    end
end 

if length(ARGS) > 1
    # this assumes sciML training set up 

    valid_trajectory = NODEData.get_trajectory(valid, 100)
    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, trajectory_call=!(TRAIN_SCIML), threshold=1e-1, modes=("average_forecast_length",), N_avg=10)
    δ = CUDA.@allowscalar grid_forecast(neural_de_sciml, ps, st)[:average_forecast_length]
    println("saving...")
    ps_save = cpu(ps)
    SlurmHyperopt.save_result(sho, HyperoptResults(pars=pars, res=δ, additonal_res=results_ad, model_pars=ps_save), i_job)
end
