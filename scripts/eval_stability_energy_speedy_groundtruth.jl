using Pkg
Pkg.activate("scripts") # assure it's the `scripts` env
    
using QG3, Plots, NeuralQG3, StatsBase, Dates, DiffEqCallbacks, JLD2, NNlib, LuxCUDA, NeuralDELux, NetCDF, ComponentArrays, SlurmHyperopt, BenchmarkTools, OrdinaryDiffEq, ParameterSchedulers, Optimisers, Random, SciMLSensitivity, Lux, Zygote, Statistics, CUDA, NODEData, Printf, EllipsisNotation

const GPU = CUDA.functional()
const DEV = NeuralQG3.DetermineDevice(gpu=GPU)

COMPUTE_DATA = false
DATA_MODEL = :speedy

NAME = "psn-speedy-200d-filter"
SAVE_DIR = "results/speedy/"

i_job = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
N_year = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

other_hyperpars = NeuralQG3.training_hyperpars()
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

train, valid, test = CUDA.@allowscalar NODEData.SingleTrajectoryBatchedOSADataloader(q, t, N_batch, valid_set=0.1, test_set=0.1)
train = NODEData.NODEDataloader_insertdim(train, 2)
test = NODEData.NODEDataloader_insertdim(test, 2)

test_trajectory = NODEData.get_trajectory(test, 300)

println("Now eval stability of ground truth:")

N_t_month = Int(floor(30 / (DT * qg3p.p.time_unit))) # 30 days
N_t_year = N_t_month*12 # 360 days 
N_year = 50

SAVE_NAME_TRUTH = string(SAVE_DIR,"Ekin-GT-Speedy-",N_year,"-2.jld2")

evolve_input = test_trajectory[2][..,end-i_job]

DT_h = NeuralQG3.QGtime_in_hours(DT, qg3p.p)
dt_speedy = NeuralQG3.compute_Δt_for_speedy(DT_h)

speedy_model = NeuralQG3.generate_speedy_data(; period_spinup=Day(90),additional_output=[:u, :v], period_data=N_year*Day(365), Δt_data=dt_speedy)
netcdf_file_path = string(speedy_model.output.run_path, "/", speedy_model.output.filename)

N_t = length(ncread(netcdf_file_path,"time"))

E_kins = zeros(Float32, 3, N_t)

velocity_unit_qg3 = qg3ppars.distance_unit / (24*60*60*qg3ppars.time_unit)
u = NetCDF.open(netcdf_file_path,"u")
v = NetCDF.open(netcdf_file_path,"v")

for i_t in 1:N_t
    u_i = u[:,:,:,i_t] ./ velocity_unit_qg3
    v_i = v[:,:,:,i_t] ./ velocity_unit_qg3
    u_i = permutedims(u_i, (3,2,1))
    v_i = permutedims(v_i, (3,2,1))
    E_kins_i = transform_SH((u_i .^ 2 + v_i .^ 2 ./ 2), qg3p)# do these things here
    E_kins[:,i_t] = E_kins_i[:,1,1]
end 

jldsave(SAVE_NAME_TRUTH; E_kins)

println("Done with Ground Truth.")





