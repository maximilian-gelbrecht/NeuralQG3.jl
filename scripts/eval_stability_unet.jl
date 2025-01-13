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

save_name_baseline_unet_node = "unet-node-sh-long-baseline"
#save_name_baseline_unet_node = "unet-node-sh-200d"
#save_name_baseline_unet_node = "unet-node-sh-era"

#NAME = "psn-speedy-200d"
NAME = "unet-qg3"

other_hyperpars = NeuralQG3.training_hyperpars()
(; N_batch, DT_FAC, data_length) = other_hyperpars

(; SAVE_NAME, SAVE_NAME_MODEL, SAVE_NAME_RESULTS, SAVE_NAME_SOL) = NeuralQG3.setup_savenames(; save_dir=SAVE_DIR, default_name=NAME)

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

test_trajectory = NODEData.get_trajectory(test, 400)

baseline_unet_node, ps_unetnode, st_unetnode = NeuralQG3.load_node_unet(save_name_baseline_unet_node, qg3p, DT; save_dir=SAVE_DIR, dev=DEV)

println("Now eval stability:")

N_blowups = 50
blowup_times = zeros(N_blowups)
evolve_input = test_trajectory[2][..,1,end:-1:end-N_blowups];

N_t_month = Int(floor(30 / (DT * qg3p.p.time_unit))) # 30 days
N_t_year = N_t_month*12 # 360 days 

evolve_input = test_trajectory[2][..,1,end:-1:end-N_blowups];

blowup_times_unet_node = zeros(N_blowups)
for i=1:N_blowups
    blowup_times_unet_node[i] = NeuralDELux.evolve_to_blowup(baseline_unet_node, ps_unetnode, st_unetnode, evolve_input[..,i], N_t=50*N_t_year)
    println("IC ",i, ": time=",blowup_times_unet_node[i])
end

JLD2.jldsave(string(SAVE_DIR, "stability_blowups-unet-",string(NAME),".jld2"); blowup_times_unet_node) 
#JLD2.jldsave(string(SAVE_DIR, "stability_blowups-unet-",string(NAME),".jld2"); stability_output_1, stability_output_2, stability_output_3) 
