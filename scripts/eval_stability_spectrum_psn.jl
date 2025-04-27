using Pkg
Pkg.activate("scripts") # assure it's the `scripts` env
    
using QG3, Plots, NeuralQG3, DiffEqCallbacks, JLD2, NNlib, LuxCUDA, NeuralDELux, ComponentArrays, SlurmHyperopt, BenchmarkTools, OrdinaryDiffEq, ParameterSchedulers, Optimisers, Random, SciMLSensitivity, Lux, Zygote, Statistics, CUDA, NODEData, Printf, EllipsisNotation

const GPU = CUDA.functional()
const DEV = NeuralQG3.DetermineDevice(gpu=GPU)

COMPUTE_DATA = true
DATA_MODEL = :qg3 # or :speedy
#DATA_MODEL = :speedy

BASELINE = false 

if DATA_MODEL == :qg3

    if BASELINE == false 
        NAME = "psn-long-swish-newhpc-final" 
    else 
        NAME = "unet-baseline-sh-qg3-2"
    end 

    SAVE_DIR = "results/qg3-long/"
elseif DATA_MODEL == :speedy

    if BASELINE == false 
        NAME = "psn-speedy-200d-sciml-filter-batch"
    else 
        NAME = "unet-baseline-sh-200d"
    end 

    SAVE_DIR = "results/speedy/"
end 


other_hyperpars = NeuralQG3.training_hyperpars()
(; N_batch, DT_FAC, data_length) = other_hyperpars

(; SAVE_NAME, SAVE_NAME_MODEL, SAVE_NAME_RESULTS, SAVE_NAME_SOL) = NeuralQG3.setup_savenames(; save_dir=SAVE_DIR, default_name=NAME)

@load SAVE_NAME_MODEL psn_hyperpars
println("Eval model with hyperpars:")
println(psn_hyperpars.pars)

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
process_based = !BASELINE
S = psn_hyperpars.pars[:additional_knowledge] ? S : nothing
neural_de_sciml, ps, st = NeuralQG3.load_psn_sciml(qg3p, psn_hyperpars, DT; alg=Tsit5(), process_based=process_based, dtmax=DT, maxiters=1e7, reltol=1e-3, SAVE_NAME=SAVE_NAME, device=DEV, S=S)
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

function average_angular_power_spectrum(A::AbstractArray{T,3}, p::QG3ModelParameters{T}) where T 
    N_t = size(A,3) 
    S = zeros(T, p.L) 
    
    for i=1:N_t 
        S += angular_power_spectrum(QG3.reorder_SH_cpu(A[:,:,i],p), p)
    end 
    S /= N_t 
    
    return S 
end 

println("Now eval stability:")

N_t_month = Int(floor(30 / (DT * qg3p.p.time_unit))) # 30 days
N_t_year = N_t_month*12 # 360 days 

if BASELINE
    MODEL_NAME = "Baseline"
else 
    MODEL_NAME = "PSN"
end 

SAVE_NAME_BASE_PSN = string(SAVE_DIR,"Spectrum-",MODEL_NAME,"-",DATA_MODEL,"-")
SAVE_NAME_TRUTH = string(SAVE_DIR,"Spectrum-GT-",DATA_MODEL,"-")

function compute_spectra(model, ps, st, ic, save_name; N=10, N_months_res=1, N_years_evolve=5)

    spectrum_input = ic 

    for i=1:N

        # evolve by five years and use at new ic for next iteration
        spectrum_input = NeuralDELux.evolve(model, ps, st, spectrum_input; N_t=N_years_evolve*N_t_year)
        spectrum_input = reshape(spectrum_input, (size(spectrum_input)..., 1))
 
        res, __ = model((range(start=0, step=DT, length=N_months_res*N_t_month), spectrum_input), ps, st)

        # compute spectra 
        S_1 = average_angular_power_spectrum(res[1,..], qg3ppars)
        S_2 = average_angular_power_spectrum(res[2,..], qg3ppars)
        S_3 = average_angular_power_spectrum(res[3,..], qg3ppars)
    
        # save spectra 
        save_name_i = string(save_name, "ev", N_years_evolve, "-m",N_months_res,"-",i,".jld2")
        jldsave(save_name_i; S_1, S_2, S_3)

        spectrum_input = res[..,end]
        
    end 
    return nothing 
end 

#PSN 
compute_spectra(neural_de_sciml, ps, st, test_trajectory[2][..,end], SAVE_NAME_BASE_PSN, N_months_res=36, N_years_evolve=0, N=20)

#GT 

println("Done with PSN, now Ground Truth...")
qg3_sciml = NeuralQG3.QG3Baseline(qg3p; dt=DT)
#compute_spectra(qg3_sciml, nothing, NamedTuple(), test_trajectory[2][..,1,end], SAVE_NAME_TRUTH, N_months_res=36, N_years_evolve=3, N=12)

println("Done with GT, finished!")






