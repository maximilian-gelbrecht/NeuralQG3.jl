using Pkg
Pkg.activate("scripts")
    
using QG3, NeuralQG3, JLD2, NNlib, LuxCUDA, NeuralDELux, Dates, ComponentArrays, SlurmHyperopt, BenchmarkTools, OrdinaryDiffEq, ParameterSchedulers, Optimisers, Random, SciMLSensitivity, Lux, Zygote, Statistics, CUDA, NODEData, Printf, EllipsisNotation

const GPU = CUDA.functional()
const DEV = NeuralQG3.DetermineDevice(gpu=GPU)

if length(ARGS) > 1
    println("Loading Hyperparameters...")
    @load "hyperopt_unet.jld2" sho 
    i_job = parse(Int,ARGS[2])
    pars = sho[i_job]  

    psn_hyperpars, other_hyperpars = NeuralQG3.parse_pars(pars)

    COMPUTE_DATA = true
    DATA_MODEL = :speedy
    TRAIN = true
    CONTINUE_TRAINING = false
    TRAIN_SCIML = true
else 
    psn_hyperpars = PSNHyperparameters(N_layers=3, additional_knowledge=true, N_Nodes=50, unet_channels=[24,48,48,96], unet_kernel=(3,3), unet_convblock=NeuralQG3.ConvBlock, activation=NNlib.swish)
    other_hyperpars = NeuralQG3.training_hyperpars(N_epoch_ad=250, N_epoch_sciml=10, N_epoch_offset=0, τ_max=6, data_length=200f0, γ=1f-5)

    COMPUTE_DATA = false
    DATA_MODEL = :speedy
    TRAIN = true
    CONTINUE_TRAINING = false
    TRAIN_SCIML = true
    i_job = nothing
end
(; DT_FAC, η, N_batch, SCALE, γ, τ_max, N_epoch_ad, N_epoch_sciml, data_length) = other_hyperpars
println("Script with Hyperpars:")
println(psn_hyperpars)
println(other_hyperpars)

#Random.seed!(123456234)
RELTOL_PREDICT = 1e-4

(; SAVE_NAME, SAVE_NAME_MODEL, SAVE_NAME_RESULTS, SAVE_NAME_RESULTS_2, SAVE_NAME_SOL) = NeuralQG3.setup_savenames(ARGS; job_id=i_job)
@save SAVE_NAME_MODEL psn_hyperpars

S, qg3ppars, ψ_0, q_0 = load_data("T42", GPU=GPU)

# pre-computations are partially performed on CPU, so we have to allow scalarindexing
qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=N_batch)
T = eltype(qg3p)
DT = T((2π/144)*DT_FAC) # in MM code: 1/144 * 2π

t_save_length = T(data_length*DT_FAC)
t_transient = T(100.)

println(qg3p)

(t, q) = NeuralQG3.get_data(DATA_MODEL, qg3p, COMPUTE_DATA; S=S, DT=DT, q_0=q_0, t_save_length=t_save_length, period_data=Day(220))

q = GPU ? T.(QG3.reorder_SH_gpu(q, qg3ppars)) : T.(q)
DT = DATA_MODEL == :era ? T(t[2]-t[1]) : DT

train, valid, test = CUDA.@allowscalar NODEData.SingleTrajectoryBatchedOSADataloader(q, t, N_batch, valid_set=0.1, test_set=0.1)
q = nothing

nn = Chain(NeuralQG3.TransformGridLayer(qg3p.g.SHtoG), WrappedFunction(identity), NeuralQG3.PermutedimLayer((2,3,1,4)), NeuralQG3.UNet(; psn_hyperpars.pars[:unet_kwargs]..., activation=psn_hyperpars.pars[:activation]), NeuralQG3.PermutedimLayer((3,1,2,4)), WrappedFunction(identity), NeuralQG3.TransformSHLayer(qg3p.g.GtoSH))
nn_single = Chain(NeuralQG3.TransformGridLayer(qg3p.g.SHtoG), WrappedFunction(x->reshape(x,size(x)...,1)),NeuralQG3.PermutedimLayer((2,3,1,4)), NeuralQG3.UNet(; psn_hyperpars.pars[:unet_kwargs]..., activation=psn_hyperpars.pars[:activation]), NeuralQG3.PermutedimLayer((3,1,2,4)), WrappedFunction(x->view(x,:,:,:,1)), NeuralQG3.TransformSHLayer(qg3p.g.GtoSH))

neural_de = ADNeuralDE(model=nn, alg=NeuralDELux.ADRK4Step(), dt=DT)
neural_de_sciml = SciMLNeuralDE(model=nn_single, alg=NeuralDELux.MultiStepRK4(), sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true), dt=DT, reltol=RELTOL_PREDICT)

rng = Random.default_rng()
ps, st = Lux.setup(rng, neural_de)
ps = NeuralQG3.gpu(DEV, ComponentArray(ps))

loss = NeuralDELux.least_square_loss_ad
loss_sciml = NeuralDELux.least_square_loss_sciml
loss_val = loss(train[1], neural_de, ps, st)

opt = Optimisers.AdamW(η, (9f-1, 9.99f-1), γ)
opt_state = Optimisers.setup(opt, ps)

η_schedule = try
    SinExp(λ0=η,λ1=1f-5,period=30,γ=0.985f0) # old syntax
catch e 
    SinExp(l0=η,l1=1f-5,period=30,decay=0.985f0) # new syntax
end

valid_trajectory = NODEData.get_trajectory(valid, 200; N_batch=N_batch)

if TRAIN 
    loss_val, gs = @time Zygote.withgradient(ps -> loss(train[1], neural_de, ps, st), ps)
    train_single_temp = (train[1][1][1,:],train[1][2][..,1,:])
    loss_val = loss_sciml(train_single_temp, neural_de_sciml, ps, st)
    loss_val, gs = @time Zygote.withgradient(ps -> loss_sciml(train_single_temp, neural_de_sciml, ps, st), ps)
end

if TRAIN
    if CONTINUE_TRAINING
        @load SAVE_NAME ps_save
        ps = gpu(ps)
    end

    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, N_forecast=175, trajectory_call=true)
    #grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, input_SH=true, output_SH=false, data_SH=false)

    neural_de, ps, st, results_ad = NeuralDELux.train!(neural_de, ps, st, loss, train, opt_state, η_schedule; N_epochs=N_epoch_ad, save_results_name=SAVE_NAME_RESULTS, shuffle_data_order=false, verbose=true, additional_metric=grid_forecast, valid_data=valid, save_name=SAVE_NAME)
    
    println("... finished AD training")
    @save SAVE_NAME_RESULTS results_ad
end 

if TRAIN_SCIML 
    train = NODEData.NODEDataloader(train, 2)
    valid = NODEData.NODEDataloader(valid, 2)
    loss_val_sciml = loss_sciml(train[1], neural_de_sciml, ps, st)

    valid_trajectory = NODEData.get_trajectory(valid, 200)
    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG,input_SH=true, N_forecast=175, output_SH=true, data_SH=true, trajectory_call=false)
    neural_de_sciml, ps, st, results_sciml = NeuralDELux.train!(neural_de_sciml, ps, st, loss_sciml, train, opt_state, η_schedule; save_results_name=SAVE_NAME_RESULTS_2, τ_range=2:τ_max, N_epochs=N_epoch_sciml, verbose=true, additional_metric=grid_forecast, valid_data=valid, scheduler_offset=N_epoch_ad, save_name=SAVE_NAME)

    # eval the forecast length
    println("...finished SciML training...")

    @save SAVE_NAME_RESULTS results_ad results_sciml
end


if length(ARGS) > 1 # saving for hyperpar optim
    valid_trajectory = NODEData.get_trajectory(valid, 100; N_batch=N_batch)
    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, trajectory_call=true, threshold=1e-1, modes=("average_forecast_length",), N_avg=10)
    δ = grid_forecast(neural_de, ps, st)[:average_forecast_length]
    println("saving...")
    ps_save = cpu(ps)
    SlurmHyperopt.save_result(sho, HyperoptResults(pars=pars, res=δ, additonal_res=results_ad, model_pars=ps_save), i_job)
end
