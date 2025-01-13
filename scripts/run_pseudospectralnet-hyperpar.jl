using Pkg
Pkg.activate("scripts")
    
using QG3, NeuralQG3, JLD2, NNlib, LuxCUDA, NeuralDELux, ComponentArrays, SlurmHyperopt, BenchmarkTools, OrdinaryDiffEq, ParameterSchedulers, Optimisers, Random, SciMLSensitivity, Lux, Zygote, Statistics, CUDA, NODEData, Printf, EllipsisNotation

const GPU = CUDA.functional()
const DEV = NeuralQG3.DetermineDevice(gpu=GPU)

if length(ARGS) > 1
    println("Loading Hyperparameters...")
    @load "hyperopt_psn_qg3_2.jld2" sho 
    i_job = parse(Int,ARGS[2])
    pars = sho[i_job]  

    psn_hyperpars, other_hyperpars = NeuralQG3.parse_pars(pars)

    PRECOMPUTE = true
    TRAIN = true
    CONTINUE_TRAINING = false
    TRAIN_SCIML = true
    SETUP_SCIML = true
else 
    psn_hyperpars = PSNHyperparameters(additional_knowledge=false, N_layers=3, N_Nodes=50, unet_channels=[24,48,96,96], unet_convblock=NeuralQG3.ConvBlock, unet_kernel=(3,3), activation=NNlib.swish)
    other_hyperpars = NeuralQG3.training_hyperpars(N_epoch_ad=0, N_epoch_sciml=4, τ_max=5, N_batch=8, N_epoch_offset=1000, data_length=200f0)

    PRECOMPUTE = true
    TRAIN = true
    CONTINUE_TRAINING = true
    TRAIN_SCIML = true
    i_job = nothing
    SETUP_SCIML = true
end
(; DT_FAC, η, N_batch, SCALE, γ, τ_max, N_epoch_ad, N_epoch_sciml, N_epoch_offset, data_length) = other_hyperpars

RELTOL_PREDICT = 1e-6
η_l1 = 1f-8
#RELTOL_PREDICT = 1e-4

(; SAVE_NAME, SAVE_NAME_MODEL, SAVE_NAME_RESULTS, SAVE_NAME_SOL) = NeuralQG3.setup_savenames(ARGS; default_name="psn-short", job_id=i_job)
if CONTINUE_TRAINING 
    println("Loading Hyperpar Config from save file...")
    @load SAVE_NAME_MODEL psn_hyperpars
else 
    @save SAVE_NAME_MODEL psn_hyperpars
end

println("Script with Hyperpars:")
println(psn_hyperpars.pars)
println(other_hyperpars)

S, qg3ppars, ψ_0, q_0 = load_data("T42", GPU=GPU)

# pre-computations are partially performed on CPU, so we have to allow scalarindexing
qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=N_batch)
T = eltype(qg3p)
DT = T((2π/144)*DT_FAC) # in MM code: 1/144 * 2π

t_save_length = T(data_length*DT_FAC)
t_transient = T(100.)

println(qg3p)

"""
if PRECOMPUTE
        t_save, sol_save = NeuralQG3.compute_QG3_data(qg3p, q_0, S, DT; t_save_length=t_save_length, t_transient=t_transient, reltol=1e-5)
        
        #@save SAVE_NAME_SOL sol_save t_save
else
        @load SAVE_NAME_SOL sol_save t_save 
end
"""
(t_save, sol_save) = NeuralQG3.get_data(:qg3, qg3p, PRECOMPUTE; S=S, DT=DT, q_0=q_0, t_save_length=t_save_length, t_transient=t_transient, reltol=1e-5)

sol_save = GPU ? T.(QG3.reorder_SH_gpu(sol_save, qg3ppars)) : T.(sol_save)

train, valid, test = CUDA.@allowscalar NODEData.SingleTrajectoryBatchedOSADataloader(sol_save, t_save, N_batch, valid_set=0.1, test_set=0.1)
println("Dataloaders:")
println("Train:")
println(size(train.data))
println("Valid:")
println(size(valid.data))
println("----------")

neural_de, ps, st = NeuralQG3.load_psn_ad(qg3p, psn_hyperpars, DT, N_batch; device=DEV)

loss = NeuralDELux.least_square_loss_ad
loss_val = loss(train[1], neural_de, ps, st)

#opt = Optimisers.AdamW(η, (9f-1, 9.99f-1), γ)
#opt = Optimisers.Lion(η, (9f-1, 9.99f-1))
opt = Optimisers.AdaBelief(η, (0.9, 0.999))
opt_state = Optimisers.setup(opt, ps)

η_schedule = try
    SinExp(λ0=η,λ1=1f-5,period=30,γ=0.985f0) # old syntax
catch e 
    SinExp(l0=η,l1=η_l1,period=50,decay=0.985f0) # new syntax

end


valid_trajectory = NODEData.get_trajectory(valid, 200; N_batch=N_batch)

if TRAIN 
    loss_val, gs = @time Zygote.withgradient(ps -> loss(train[1], neural_de, ps, st)[1], ps)
end

if CONTINUE_TRAINING
    ps = NeuralQG3.load_ps(SAVE_NAME, DEV)
end

if TRAIN
    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, N_forecast=175, trajectory_call=true)
    neural_de, ps, st, results_ad = NeuralDELux.train!(neural_de, ps, st, loss, train, opt_state, η_schedule; τ_range=2:2, N_epochs=N_epoch_ad, shuffle_data_order=false, verbose=true, additional_metric=grid_forecast, scheduler_offset=N_epoch_offset, valid_data=valid, save_name=SAVE_NAME)
    #neural_de, ps, st, results_ad = NeuralDELux.train!(neural_de, ps, st, loss, train, opt_state, η_schedule; τ_range=2:2, N_epochs=N_epoch_ad, shuffle_data_order=false, verbose=true, scheduler_offset=N_epoch_offset, save_name=SAVE_NAME, valid_data=valid)

    println("... finished AD training")
    @save SAVE_NAME_RESULTS results_ad
end 

if SETUP_SCIML 
    N_batch_sciml = N_batch

    qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=N_batch_sciml)
    
    #neural_de_sciml, __, __ = NeuralQG3.load_psn_sciml(qg3p, psn_hyperpars, DT; N_batch=N_batch_sciml, device=DEV, init=false, S=S, alg=NeuralDELux.MultiStepRK4(), RELTOL_PREDICT=RELTOL_PREDICT)
    neural_de_sciml, __, __ = NeuralQG3.load_psn_sciml(qg3p, psn_hyperpars, DT; N_batch=N_batch_sciml, device=DEV, init=false, S=S, alg=Tsit5(), RELTOL_PREDICT=RELTOL_PREDICT)

    loss_sciml = CUDA.@allowscalar NeuralQG3.SpectralFilteringLoss(qg3p, 2:τ_max; N_batch=N_batch_sciml)

    #loss_sciml = NeuralDELux.least_square_loss_sciml

    #train = NODEData.NODEDataloader_insertdim(train, 2)
    #valid = NODEData.NODEDataloader_insertdim(valid, 2)
    #loss_val_sciml = loss_sciml(train[1], neural_de_sciml, ps, st)

    #valid_trajectory = NODEData.get_trajectory(valid, 100)

    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG,input_SH=true, output_SH=true, data_SH=true, N_forecast=175, trajectory_call=false)

    if TRAIN_SCIML 

        begin  
            #valid_3 = remake_dataloader(valid, 7) 
            #valid_3 = valid_3[1]
            valid_3 = valid[1]
            #loss_val, gs = CUDA.@time Zygote.withgradient(ps -> loss_sciml(valid_3, neural_de_sciml, ps, st)[1], ps);
            loss_val, gs = CUDA.@time Zygote.withgradient(ps -> loss_sciml[1](valid_3, neural_de_sciml, ps, st)[1], ps);
        end 

        neural_de_sciml, ps, st, results_sciml = NeuralDELux.train!(neural_de_sciml, ps, st, loss_sciml, train, opt_state, η_schedule; save_mode=:additional_metric, τ_range=2:τ_max, N_epochs=N_epoch_sciml, verbose=true, additional_metric=grid_forecast, valid_data=valid, scheduler_offset=N_epoch_ad + N_epoch_offset, save_name=SAVE_NAME)
        #neural_de_sciml, ps, st, results_sciml = NeuralDELux.train!(neural_de_sciml, ps, st, loss_sciml, train, opt_state, η_schedule; τ_range=2:τ_max, N_epochs=N_epoch_sciml, verbose=true, scheduler_offset=N_epoch_ad + N_epoch_offset, save_name=SAVE_NAME, valid_data=valid)

        # eval the forecast length
        println("...finished SciML training...")
    
        @save SAVE_NAME_RESULTS results_ad results_sciml
    end
end 

if length(ARGS) > 1 # saves final results for hyperpar optimziation
    # this assumes sciML training set up 

    valid_trajectory = NODEData.get_trajectory(valid, 100)

    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, trajectory_call=!(TRAIN_SCIML), threshold=1e-1, modes=("average_forecast_length",), N_avg=10)
    δ = CUDA.@allowscalar grid_forecast(neural_de_sciml, ps, st)[:average_forecast_length]
    println("saving...")
    ps_save = cpu(ps)
    if TRAIN_SCIML
        SlurmHyperopt.save_result(sho, HyperoptResults(pars=pars, res=δ, additonal_res=results_sciml, model_pars=ps_save), i_job)
    else 
        SlurmHyperopt.save_result(sho, HyperoptResults(pars=pars, res=δ, additonal_res=results_ad, model_pars=ps_save), i_job)
    end 
    println("saving...")
end
