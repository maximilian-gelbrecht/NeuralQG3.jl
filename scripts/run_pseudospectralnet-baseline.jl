using Pkg
Pkg.activate("scripts/")

using QG3, NeuralQG3, JLD2, NNlib, Dates, LuxCUDA, NeuralDELux, ComponentArrays, SlurmHyperopt, BenchmarkTools, OrdinaryDiffEq, ParameterSchedulers, Optimisers, Random, SciMLSensitivity, Lux, Zygote, Statistics, CUDA, NODEData, Printf, EllipsisNotation

const GPU = CUDA.functional()
const DEV = NeuralQG3.DetermineDevice(gpu=GPU)

if length(ARGS) > 1
    println("Loading Hyperparameters...")
    @load "psn_hyperpar_speedy_newhpc.jld2" sho 
    i_job = parse(Int,ARGS[2])
    pars = sho[i_job]  

    psn_hyperpars, other_hyperpars = NeuralQG3.parse_pars(pars; additional_knowledge=true)

    PRECOMPUTE = false
    PRETRAIN = false
    TRAIN = true
    CONTINUE_TRAINING = false
    SETUP_SCIML = true
    TRAIN_SCIML = false
    DATA_MODEL = :qg3
else 
    psn_hyperpars = PSNHyperparameters(N_layers=3, additional_knowledge=false, N_Nodes=50, unet_channels=[24,48,48,96], unet_kernel=(3,3), unet_convblock=NeuralQG3.LongLatConvBlock, activation=NNlib.swish)
    other_hyperpars = NeuralQG3.training_hyperpars(N_epoch_ad=200, N_epoch_sciml=10, N_batch=14, N_epoch_offset=0, τ_max=6, data_length=200f0, γ=1f-5)

    PRECOMPUTE = true
    PRETRAIN = true
    TRAIN = true
    CONTINUE_TRAINING = false
    SETUP_SCIML = true
    TRAIN_SCIML = true
    i_job = nothing
    DATA_MODEL = :qg3
end
(; DT_FAC, η, N_batch, SCALE, γ, τ_max, N_epoch_ad, N_epoch_sciml, N_epoch_offset, data_length) = other_hyperpars


#Random.seed!(123456234) #123456
RELTOL_PREDICT = 1e-3

(; SAVE_NAME, SAVE_NAME_MODEL, SAVE_NAME_RESULTS, SAVE_NAME_RESULTS_2, SAVE_NAME_SOL) = NeuralQG3.setup_savenames(ARGS; job_id=i_job, default_name="psn-speedy-new-longer")
if CONTINUE_TRAINING 
    println("Loading Hyperpar Config from save file...")
    @load SAVE_NAME_MODEL psn_hyperpars
else 
    @save SAVE_NAME_MODEL psn_hyperpars
end

println(psn_hyperpars)
println(other_hyperpars)

# load QG3 model components 
S, qg3ppars, ψ_0, q_0 = load_data("T42", GPU=GPU)


# pre-computations are partially performed on CPU, so we have to allow scalarindexing
qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=N_batch)

T = eltype(qg3p)
DT = T(2π/144*DT_FAC)
data_length = T(data_length*DT_FAC)

(t, q) = NeuralQG3.get_data(DATA_MODEL, qg3p, PRECOMPUTE; S=S, DT=DT, t_transient=T(100.), q_0=q_0, period_data=Day(220), t_save_length=data_length)
q = GPU ? T.(QG3.reorder_SH_gpu(q, qg3ppars)) : T.(q)

train, valid, test = CUDA.@allowscalar NODEData.SingleTrajectoryBatchedOSADataloader(q, t, N_batch, valid_set=0.1, test_set=0.1)

println("Dataloaders:")
println("Train:")
println(size(train.data))
println("Valid:")
println(size(valid.data))
println("----------")

neural_de, ps, st = NeuralQG3.load_psn_ad(qg3p, psn_hyperpars, DT, N_batch; device=DEV, S=S, process_based=false)

if CONTINUE_TRAINING
    ps = NeuralQG3.load_ps(SAVE_NAME, DEV)
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

#valid_trajectory = NODEData.get_trajectory(valid, 200; N_batch=N_batch)

if TRAIN 
    train_1 = train[1]
    loss_val, gs = @time Zygote.withgradient(ps -> loss(train_1, neural_de, ps, st)[1], ps)
end

if TRAIN
    #grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, N_forecast=175, trajectory_call=true)

    #neural_de, ps, st, results_ad = CUDA.@time NeuralDELux.train!(neural_de, ps, st, loss, train, opt_state, η_schedule; save_mode=:additional_metric, τ_range=2:2, N_epochs=N_epoch_ad, save_results_name=SAVE_NAME_RESULTS, shuffle_data_order=false, verbose=true, additional_metric=grid_forecast, valid_data=valid, save_name=SAVE_NAME, scheduler_offset=N_epoch_offset)
    neural_de, ps, st, results_ad = CUDA.@time NeuralDELux.train!(neural_de, ps, st, loss, train, opt_state, η_schedule; τ_range=2:2, N_epochs=N_epoch_ad, save_results_name=SAVE_NAME_RESULTS, shuffle_data_order=false, verbose=true, valid_data=valid, save_name=SAVE_NAME, scheduler_offset=N_epoch_offset)

    println("... finished AD training")
    @save SAVE_NAME_RESULTS results_ad
end

if SETUP_SCIML 
    # further epochs with Tsit single (it's more accurate)
    N_batch_sciml = N_batch
    qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=N_batch_sciml)
    
    #neural_de_sciml, __, __ = NeuralQG3.load_psn_sciml(qg3p, psn_hyperpars, DT; N_batch=N_batch_sciml, device=DEV, init=false, S=S, alg=Tsit5(), sensealg=GaussAdjoint(autojacvec=ZygoteVJP()), RELTOL_PREDICT=RELTOL_PREDICT)
    neural_de_sciml, __, __ = NeuralQG3.load_psn_sciml(qg3p, psn_hyperpars, DT; N_batch=N_batch_sciml, device=DEV, init=false, S=S, alg=NeuralDELux.MultiStepRK4(), RELTOL_PREDICT=RELTOL_PREDICT, process_based=false)

    #loss_sciml = NeuralDELux.least_square_loss_sciml
    loss_sciml = CUDA.@allowscalar NeuralQG3.SpectralFilteringLoss(qg3p, 2:τ_max; N_batch=N_batch_sciml)
    #train = NODEData.NODEDataloader_insertdim(train, 2)
    #valid = NODEData.NODEDataloader_insertdim(valid, 2)
    #loss_val_sciml = loss_sciml[1](train[1], neural_de_sciml, ps, st)
    #loss_val_sciml = loss_sciml(train[1], neural_de_sciml, ps, st)

    #valid_trajectory = NeuralQG3.trajectory_insert_batchdim(NODEData.get_trajectory(valid, 200); N_batch)
    #valid_trajectory = NODEData.get_trajectory(valid, 200)

    #grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, N_forecast=175, input_SH=true, output_SH=true, data_SH=true, trajectory_call=false)

    if TRAIN_SCIML
        # precompile
        begin  
            #valid_3 = remake_dataloader(valid, 7) 
            #valid_3 = valid_3[1]
            valid_3 = valid[1]
            #loss_val, gs = CUDA.@time Zygote.withgradient(ps -> loss_sciml(valid_3, neural_de_sciml, ps, st)[1], ps);
            loss_val, gs = CUDA.@time Zygote.withgradient(ps -> loss_sciml[1](valid_3, neural_de_sciml, ps, st)[1], ps);
        end 

        #neural_de_sciml, ps, st, results_sciml = CUDA.@time NeuralDELux.train!(neural_de_sciml, ps, st, loss_sciml, train, opt_state, η_schedule; save_mode=:additional_metric, τ_range=2:τ_max, N_epochs=N_epoch_sciml, verbose=true, additional_metric=grid_forecast, save_results_name=SAVE_NAME_RESULTS_2, valid_data=valid, scheduler_offset=N_epoch_ad + N_epoch_offset, save_name=SAVE_NAME)
        neural_de_sciml, ps, st, results_sciml = CUDA.@time NeuralDELux.train!(neural_de_sciml, ps, st, loss_sciml, train, opt_state, η_schedule; τ_range=2:τ_max, N_epochs=N_epoch_sciml, verbose=true, save_results_name=SAVE_NAME_RESULTS_2, valid_data=valid, scheduler_offset=N_epoch_ad + N_epoch_offset, save_name=SAVE_NAME)

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

 