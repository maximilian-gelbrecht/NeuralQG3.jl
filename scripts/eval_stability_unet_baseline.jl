using Pkg
Pkg.activate("scripts") # assure it's the `scripts` env
    
using QG3, Plots, NeuralQG3, JLD2, NNlib, LuxCUDA, NeuralDELux, ComponentArrays, SlurmHyperopt, BenchmarkTools, OrdinaryDiffEq, ParameterSchedulers, Optimisers, Random, SciMLSensitivity, Lux, Zygote, Statistics, CUDA, NODEData, Printf, EllipsisNotation

const GPU = CUDA.functional()
const DEV = NeuralQG3.DetermineDevice(gpu=GPU)

COMPUTE_DATA = true
DATA_MODEL = :qg3 # or :speedy
#DATA_MODEL = :speedy
#DATA_MODEL = :era
SAVE_DIR = "results/qg3-long/"
#SAVE_DIR = ""
#SAVE_DIR = "results/speedy/"
#SAVE_DIR = "results/era/"

#save_name_baseline_unet_node = "unet-node-sh-long-baseline"
#save_name_baseline_unet_node = "unet-node-sh-200d"
#save_name_baseline_unet_node = "unet-node-sh-era"
#save_name_baseline_unet_node = "unet-baseline-sh-200d"

save_name_baseline_unet_node = "unet-baseline-sh-qg3"

#NAME = "unet-baseline-sh-200d"
NAME = "unet-baseline-sh-qg3"

#NAME = "unet-qg3"

i_job = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0

other_hyperpars = NeuralQG3.training_hyperpars()
(; N_batch, DT_FAC, data_length) = other_hyperpars

(; SAVE_NAME, SAVE_NAME_MODEL, SAVE_NAME_RESULTS, SAVE_NAME_SOL) = NeuralQG3.setup_savenames(; save_dir=SAVE_DIR, default_name=NAME)


@load SAVE_NAME_MODEL psn_hyperpars
println("Eval model with hyperpars:")
println(psn_hyperpars.pars)

S, qg3ppars, ψ_0, q_0 = load_data("T42", GPU=GPU)
qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=1)
T = eltype(qg3p)
DT = T((2π/144)*DT_FAC) # in MM code: 1/144 * 2π
data_length = T(data_length*DT_FAC)

println(qg3p)

(t, q) = NeuralQG3.get_data(DATA_MODEL, qg3p, COMPUTE_DATA; S=S, DT=DT, q_0=q_0, t_save_length=data_length)
DT = DATA_MODEL == :era ? T(t[2]-t[1]) : DT

q = GPU ? T.(QG3.reorder_SH_gpu(q, qg3ppars)) : T.(q)

train, valid, test = CUDA.@allowscalar NODEData.SingleTrajectoryBatchedOSADataloader(q, t, N_batch, valid_set=0.1, test_set=0.1)
train = NODEData.NODEDataloader_insertdim(train, 2)
test = NODEData.NODEDataloader_insertdim(test, 2)


S = psn_hyperpars.pars[:additional_knowledge] ? S : nothing
baseline_unet_node, ps, st = NeuralQG3.load_psn_sciml(qg3p, psn_hyperpars, DT; process_based=false, SAVE_NAME=SAVE_NAME, device=DEV, S=S, dtmax=DT, maxiters=1e6)
test_trajectory = NODEData.get_trajectory(test, 300)
grid_forecast = NeuralQG3.GridForecast(test_trajectory, qg3p.g.SHtoG,input_SH=true, output_SH=true, data_SH=true, trajectory_call=false)
gf = grid_forecast(baseline_unet_node, ps, st)

println("Now eval stability:")

N_blowups = 1
blowup_times = zeros(N_blowups)
evolve_input = test_trajectory[2][..,1,end:-1:end-N_blowups];

N_t_month = Int(floor(30 / (DT * qg3p.p.time_unit))) # 30 days
N_t_year = N_t_month*12 # 360 days 

if i_job == 0 # all in one job 
    evolve_input = test_trajectory[2][..,end:-1:end-N_blowups];

    blowup_times_unet_node = zeros(N_blowups)
    for i=1:N_blowups
        blowup_times_unet_node[i] = NeuralDELux.evolve_to_blowup(baseline_unet_node, ps, st, evolve_input[..,i], N_t=10*N_t_year)
        println("IC ",i, ": time=",blowup_times_unet_node[i])
    end
    JLD2.jldsave(string(SAVE_DIR, "stability_blowups-unet-",string(NAME),".jld2"); blowup_times_unet_node) 
else 
    N_year = 50
    println("N_year= ",N_year)
    evolve_input = test_trajectory[2][..,i_job]
    blowuptime = NeuralDELux.evolve_to_blowup(baseline_unet_node, ps, st, evolve_input, N_t=N_year*N_t_year) 

    println("Blowup at: ",blowuptime)
end 

#JLD2.jldsave(string(SAVE_DIR, "stability_blowups-unet-",string(NAME),".jld2"); stability_output_1, stability_output_2, stability_output_3) 
