using Pkg
Pkg.activate("scripts") # assure it's the `scripts` env
    
using QG3, Plots, NeuralQG3, DiffEqCallbacks, JLD2, NNlib, LuxCUDA, NeuralDELux, ComponentArrays, SlurmHyperopt, BenchmarkTools, OrdinaryDiffEq, ParameterSchedulers, Optimisers, Random, SciMLSensitivity, Lux, Zygote, Statistics, CUDA, NODEData, Printf, EllipsisNotation

const GPU = CUDA.functional()
const DEV = NeuralQG3.DetermineDevice(gpu=GPU)

COMPUTE_DATA = true
DATA_MODEL = :qg3 # or :speedy
#DATA_MODEL = :speedy
NAME = "psn-long-swish-newhpc-final" 
#NAME = "psn-speedy-200d-sciml-filter-batch"

#SAVE_DIR = ""
SAVE_DIR = "results/qg3-long/"
#SAVE_DIR = "results/speedy/"

i_job = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
N_year = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

other_hyperpars = NeuralQG3.training_hyperpars()
(; N_batch, DT_FAC, data_length) = other_hyperpars

(; SAVE_NAME, SAVE_NAME_MODEL, SAVE_NAME_RESULTS, SAVE_NAME_SOL) = NeuralQG3.setup_savenames(; save_dir=SAVE_DIR, default_name=NAME)

@load SAVE_NAME_MODEL psn_hyperpars
println("Eval model with hyperpars:")
println(psn_hyperpars.pars)
println("Eval trajectory no.=", i_job)

# load process-based core
S, qg3ppars, ψ_0, q_0 = load_data("T42", GPU=GPU)
qg3p = CUDA.@allowscalar QG3Model(qg3ppars; N_batch=1)
T = eltype(qg3p)
DT = T((2π/144)*DT_FAC) # in MM code: 1/144 * 2π
data_length = T(data_length*DT_FAC)

println(qg3p)

# load data
(t, q) = NeuralQG3.get_data(DATA_MODEL, qg3p, COMPUTE_DATA; S=S, DT=DT, q_0=q_0, t_save_length=data_length)
DT = DATA_MODEL == :era ? T(t[2]-t[1]) : DT

q = GPU ? T.(QG3.reorder_SH_gpu(q, qg3ppars)) : T.(q)
train, valid , test= CUDA.@allowscalar NODEData.SingleTrajectoryBatchedOSADataloader(q, t, N_batch, valid_set=0.1, test_set=0.1)
train = NODEData.NODEDataloader_insertdim(train, 2)
test = NODEData.NODEDataloader_insertdim(test, 2)

# load PSN 
S = psn_hyperpars.pars[:additional_knowledge] ? S : nothing
neural_de_sciml, ps, st = NeuralQG3.load_psn_sciml(qg3p, psn_hyperpars, DT; alg=Tsit5(), dtmax=DT, maxiters=1e7, reltol=1e-3, SAVE_NAME=SAVE_NAME, device=DEV, S=S)
test_trajectory = NODEData.get_trajectory(test, 300)

# setup forecast task 
grid_forecast = NeuralQG3.GridForecast(test_trajectory, qg3p.g.SHtoG,input_SH=true, output_SH=true, data_SH=true, trajectory_call=false)
gf = grid_forecast(neural_de_sciml, ps, st)

println(gf)

function angular_power_spectrum(A::AbstractArray{T,2}, p::QG3ModelParameters{T}) where T
    
    S = zeros(T, p.L)
    for l = 0:(p.L-1)
        fac = 1/(2l + 1)

        S_l = 0 
        for m ∈ -l:l
            im = m<0 ? 2*abs(m) : 2*m+1
            il = l + 1 - abs(m)
            S_l += A[il,im]*A[il,im]
        end 
        S[l+1] = fac * S_l
    end 
    return S 
end 

function average_angular_power_spectrum(A::AbstractArray{T,4}, p::QG3ModelParameters{T}) where T 
    N_t = size(A,4) 
    S = zeros(T, p.L) 
    
    for i=1:N_t 
        S += angular_power_spectrum(QG3.reorder_SH_cpu(A[:,:,1,i],p), p)
    end 
    S /= N_t 
    
    return S 
end 

println("Now eval stability:")

N_t_month = Int(floor(30 / (DT * qg3p.p.time_unit))) # 30 days
N_t_year = N_t_month*12 # 360 days 

SAVE_NAME_BASE_PSN = string(SAVE_DIR,"Spectrum-Month-PSN-")
SAVE_NAME_TRUTH = string(SAVE_DIR,"Spectrum-Month-GT-")

spectrum_input = test_trajectory[2][..,i_job:i_job]

# PSN

for i=1:10
    # integrate dense solution 
    res, st = neural_de_sciml((range(start=0, step=DT, length=N_t_month), spectrum_input), ps, st)

    # compute spectra 
    S_1 = average_angular_power_spectrum(res[1,..], qg3ppars)
    S_2 = average_angular_power_spectrum(res[2,..], qg3ppars)
    S_3 = average_angular_power_spectrum(res[3,..], qg3ppars)

    # save spectra 
    save_name_i = string(SAVE_NAME_BASE_PSN, "-",i,".jld2")
    jldsave(save_name_i; S_1, S_2, S_3)

    # evolve by five years and use at new ic for next iteration
    spectrum_input = NeuralDELux.evolve(neural_de_sciml, ps, st, res[..,end]; N_t=5*N_t_year)
    spectrum_input = reshape(spectrum_input, (size(spectrum_input)..., 1))
end 

println("Done with PSN, now Ground Truth...")
qg3_sciml = NeuralQG3.QG3Baseline(qg3p; dt=DT)

for i=1:10
    # integrate dense solution 
    res, st = qg3_sciml((range(start=0, step=DT, length=N_t_month), spectrum_input), nothing, NamedTuple())

    # compute spectra 
    S_1 = average_angular_power_spectrum(res[1,..], qg3ppars)
    S_2 = average_angular_power_spectrum(res[2,..], qg3ppars)
    S_3 = average_angular_power_spectrum(res[3,..], qg3ppars)

    # save spectra 
    save_name_i = string(SAVE_NAME_BASE_TRUTH, "-",i,".jld2")
    jldsave(save_name_i; S_1, S_2, S_3)

    # evolve by five years and use at new ic for next iteration
    spectrum_input = NeuralDELux.evolve(qg3_sciml, ps, st, res[..,end]; N_t=5*N_t_year)
    spectrum_input = reshape(spectrum_input, (size(spectrum_input)..., 1))
end 

println("Done with GT, finished!")






