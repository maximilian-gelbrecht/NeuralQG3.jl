using Pkg
Pkg.activate("scripts")
    
using QG3, NeuralQG3, JLD2, NNlib, LuxCUDA, Dates, NeuralDELux, ComponentArrays, SlurmHyperopt, BenchmarkTools, OrdinaryDiffEq, ParameterSchedulers, Optimisers, Random, SciMLSensitivity, Lux, Zygote, Statistics, CUDA, NODEData, Printf, EllipsisNotation

const GPU = CUDA.functional()
const DEV = NeuralQG3.DetermineDevice(gpu=GPU)

if length(ARGS) > 1
    println("Loading Hyperparameters...")
    @load "hyperopt_unet.jld2" sho 
    i_job = parse(Int,ARGS[2])
    pars = sho[i_job]  

    psn_hyperpars, other_hyperpars = NeuralQG3.parse_pars(pars)

    COMPUTE_DATA = true
    DATA_MODEL = :qg3
    TRAIN = true
    CONTINUE_TRAINING = false
else 
    psn_hyperpars = PSNHyperparameters(N_layers=3, additional_knowledge=true, N_Nodes=50, unet_channels=[24,48,48,96], unet_kernel=(3,3), unet_convblock=NeuralQG3.LongLatConvBlock, activation=NNlib.swish)
    other_hyperpars = NeuralQG3.training_hyperpars(N_epoch_ad=250, N_epoch_sciml=3, N_epoch_offset=0, τ_max=6, data_length=200f0, γ=1f-5)

    COMPUTE_DATA = false
    DATA_MODEL = :qg3
    TRAIN = true
    CONTINUE_TRAINING = false
    i_job = nothing
end

(; DT_FAC, η, N_batch, SCALE, γ, τ_max, N_epoch_ad, N_epoch_sciml, data_length) = other_hyperpars
println("Script with Hyperpars:")
println(psn_hyperpars)
println(other_hyperpars)

#Random.seed!(345345345)
RELTOL_PREDICT = 1e-4

(; SAVE_NAME, SAVE_NAME_MODEL, SAVE_NAME_RESULTS, SAVE_NAME_SOL) = NeuralQG3.setup_savenames(ARGS; job_id=i_job)
@save SAVE_NAME_MODEL psn_hyperpars

S, qg3ppars, ψ_0, q_0 = load_data("T42", GPU=GPU)

# pre-computations are partially performed on CPU, so we have to allow scalarindexing
qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=N_batch)
T = eltype(qg3p)
DT = T((2π/144)*DT_FAC) # in MM code: 1/144 * 2π

t_save_length = T(T(200.)*DT_FAC)
t_transient = T(100.)
t_end = t_transient + t_save_length

println(qg3p)

(t, q) = NeuralQG3.get_data(DATA_MODEL, qg3p, COMPUTE_DATA; S=S, DT=DT, q_0=q_0, t_save_length=t_save_length, period_data=Day(365))
DT = DATA_MODEL == :era ? T(t[2]-t[1]) : DT

q = GPU ? T.(QG3.reorder_SH_gpu(q, qg3ppars)) : T.(q)

# transform to grid 
q = QG3.transform_grid_data(q, qg3p)

# prepare training data
train, valid, test = CUDA.@allowscalar NODEData.SingleTrajectoryBatchedOSADataloader(q, t, N_batch, valid_set=0.1, test_set=0.1)
q = nothing

nn = Chain(NeuralQG3.PermutedimLayer((2,3,1,4)), NeuralQG3.UNet(; psn_hyperpars.pars[:unet_kwargs]..., activation=psn_hyperpars.pars[:activation]), NeuralQG3.PermutedimLayer((3,1,2,4)))

RELTOL_PREDICT = 1f-3
neural_de = NeuralQG3.RecursiveNet(nn)

rng = Random.default_rng()
ps, st = Lux.setup(rng, neural_de)
ps = NeuralQG3.gpu(DEV, ComponentArray(ps))

loss = NeuralDELux.least_square_loss_ad

loss_val = loss(train[1], neural_de, ps, st)

opt = Optimisers.AdamW(η, (9f-1, 9.99f-1), γ)
opt_state = Optimisers.setup(opt, ps)

η_schedule = try
    SinExp(λ0=η,λ1=1f-5,period=30,γ=0.985f0) # old syntax
catch e 
    SinExp(l0=η,l1=1f-5,period=30,decay=0.985f0) # new syntax
end
valid_trajectory = NODEData.get_trajectory(valid, 100; N_batch=N_batch) # do a new get trajectory for this in NODEData.jl

if TRAIN 
    loss_val, gs = @time Zygote.withgradient(ps -> loss(train[1], neural_de, ps, st)[1], ps)
end

if CONTINUE_TRAINING
    @load SAVE_NAME_MODEL ps_save
    ps = gpu(ps)
end

if TRAIN
    # first epoch with Euler (it's faster)
    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG; trajectory_call=true, input_SH=false, data_SH=false, output_SH=false)
    neural_de, ps, st, results = NeuralDELux.train!(neural_de, ps, st, loss, train, opt_state, η_schedule; N_epochs=N_epoch_ad, shuffle_data_order=false, verbose=true, additional_metric=grid_forecast, save_name=SAVE_NAME, save_results_name=SAVE_NAME_RESULTS)
    
    @save SAVE_NAME_RESULTS results 

    # eval the forecast length
    println("...finished training...")
    grid_forecast = NeuralQG3.remake(grid_forecast; modes=("average_forecast_length",))
    δ = grid_forecast(neural_de, ps, st)
    println("forecast length (e=0.1) = ",δ)
end

if length(ARGS) > 1 # saving for hyperpar optim
    valid_trajectory = NODEData.get_trajectory(valid, 100; N_batch=N_batch)
    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, trajectory_call=true, threshold=1e-1, modes=("average_forecast_length",), N_avg=10)
    δ = grid_forecast(neural_de, ps, st)[:average_forecast_length]
    println("saving...")
    ps_save = cpu(ps)
    SlurmHyperopt.save_result(sho, HyperoptResults(pars=pars, res=δ, additonal_res=results_ad, model_pars=ps_save), i_job)
end

