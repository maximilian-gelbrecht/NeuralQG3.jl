using Pkg
Pkg.activate("scripts") # assure it's the `scripts` env
    
using QG3, Plots, NeuralQG3, DiffEqCallbacks, JLD2, NNlib, LuxCUDA, NeuralDELux, ComponentArrays, SlurmHyperopt, BenchmarkTools, OrdinaryDiffEq, ParameterSchedulers, Optimisers, Random, SciMLSensitivity, Lux, Zygote, Statistics, CUDA, NODEData, Printf, EllipsisNotation

const GPU = CUDA.functional()
const DEV = NeuralQG3.DetermineDevice(gpu=GPU)

COMPUTE_DATA = true

#DATA_MODEL = :qg3 # or :speedy
DATA_MODEL = :speedy

if DATA_MODEL == :qg3
    NAME = "unet-baseline-sh-qg3"
    SAVE_DIR = "results/qg3-long/"
elseif DATA_MODEL == :speedy
    NAME = "unet-baseline-sh-200d"
    SAVE_DIR = "results/speedy/"
end 

i_job = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
N_year = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

other_hyperpars = NeuralQG3.training_hyperpars(data_length=200)
(; N_batch, DT_FAC, data_length) = other_hyperpars

(; SAVE_NAME, SAVE_NAME_MODEL, SAVE_NAME_RESULTS, SAVE_NAME_SOL) = NeuralQG3.setup_savenames(; save_dir=SAVE_DIR, default_name=NAME)

@load SAVE_NAME_MODEL psn_hyperpars
println("Eval model with hyperpars:")
println(psn_hyperpars.pars)
println("Eval trajectory no.=", i_job)


S, qg3ppars, ψ_0, q_0 = load_data("T42", GPU=GPU)
qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=1)
T = eltype(qg3p)
DT = T((2π/144)*DT_FAC) # in MM code: 1/144 * 2π
data_length = T(data_length*DT_FAC)

println(qg3p)

(t, q) = NeuralQG3.get_data(DATA_MODEL, qg3p, COMPUTE_DATA; S=S, DT=DT, q_0=q_0, t_save_length=data_length)
DT = DATA_MODEL == :era ? T(t[2]-t[1]) : DT

q = GPU ? T.(QG3.reorder_SH_gpu(q, qg3ppars)) : T.(q)
#train, valid = NODEData.NODEDataloader(reshape(q,size(q,1),size(q,2),size(q,3),1,size(q,4)), t, 2, valid_set=0.1)
train, valid, test = CUDA.@allowscalar NODEData.SingleTrajectoryBatchedOSADataloader(q, t, N_batch, valid_set=0.1, test_set=0.1)
train = NODEData.NODEDataloader_insertdim(train, 2)
test = NODEData.NODEDataloader_insertdim(test, 2)

S = psn_hyperpars.pars[:additional_knowledge] ? S : nothing

neural_de_sciml, ps_unetnode, st_unetnode = NeuralQG3.load_psn_sciml(qg3p, psn_hyperpars, DT; process_based=false, SAVE_NAME=SAVE_NAME, device=DEV, S=S, dtmax=DT, maxiters=1e6)

test_trajectory = NODEData.get_trajectory(test, 300)
grid_forecast = NeuralQG3.GridForecast(test_trajectory, qg3p.g.SHtoG,input_SH=true, output_SH=true, data_SH=true, trajectory_call=false)
gf = grid_forecast(neural_de_sciml, ps_unetnode, st_unetnode)

println(gf)

println("Now eval stability:")

N_t_month = Int(floor(30 / (DT * qg3p.p.time_unit))) # 30 days
N_t_year = N_t_month*12 # 360 days 

SAVE_NAME_TRUTH = string(SAVE_DIR,"Ekin-GT-",N_year,"y-",NAME,"-i",i_job,".jld2")
SAVE_NAME_PSN = string(SAVE_DIR,"Ekin-UNET-",N_year,"y-",NAME,"-i",i_job,".jld2")

evolve_input = test_trajectory[2][..,end-i_job]

vals_qg3 = SavedValues(T,CuArray{T,1})
qg3_sciml = NeuralQG3.QG3Baseline(qg3p; dt=DT)
qg3_sol = NeuralDELux.evolve_sol(qg3_sciml, nothing, NamedTuple(), evolve_input[:,:,:,1]; dt=DT, N_t=N_year*N_t_year, callback=SavingCallback(QG3.KineticEnergyCallback(qg3p), vals_qg3, saveat=20*DT)) 
vals_qg3 = Matrix(hcat(vals_qg3.saveval...))
jldsave(SAVE_NAME_TRUTH; vals_qg3) 

println("Done with Ground Truth, now UNET...")

vals_psn = SavedValues(T,CuArray{T,2})
psn_sol = NeuralDELux.evolve_sol(neural_de_sciml, ps_unetnode, st_unetnode, evolve_input; dt=DT, N_t=N_year*N_t_year, callback=SavingCallback(QG3.KineticEnergyCallback(qg3p), vals_psn, saveat=20*DT)) #long_evolution_snap_g = transform_grid(long_evolution_snap, qg3p);
vals_psn = Array(hcat(vals_psn.saveval...))
jldsave(SAVE_NAME_PSN; vals_psn) 

println("Done with PSN, finished!")



