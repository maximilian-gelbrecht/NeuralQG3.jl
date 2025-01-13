import Pkg 
Pkg.activate("scripts")

using Plots, JLD2, StatsBase, NeuralQG3, QG3

__, qg3ppars, __, __ = load_data("T42", GPU=false)

EXPERIMENT = :speedy # :speedy 

if EXPERIMENT == :qg3 
    ground_truth_energy = load("results/qg3-long/Ekin-QG3-GroundTruth.jld2")
    ground_truth_energy = ground_truth_energy["vals_qg3"]

    psn_energy = load("results/qg3-long/Ekin-QG3-PSN.jld2")
    unet_energy = load("results/qg3-long/Ekin-QG3-UNET-SH-Baseline.jld2")
elseif EXPERIMENT == :speedy 
    ground_truth_energy = load("results/speedy/Ekin-GT-Speedy-50.jld2")
    ground_truth_energy = ground_truth_energy["E_kins"]

    # we sampled the GT for speedy at a x20 higher rate
    ground_truth_energy = ground_truth_energy[:,1:20:end]

    psn_energy = load("results/speedy/Ekin-PSN-50y-psn-speedy-200d-filter.jld2")
    unet_energy = load("results/speedy/Ekin-UNET-50y-unet-baseline-sh-200d.jld2")
end 

Δt_d = 20*(((2π/144)*10)) * qg3ppars.time_unit # [d] # 20 (saving cb) * DT * conversion (days) * conversion (years)

psn_energy = psn_energy["vals_psn"]
unet_energy = unet_energy["vals_psn"]

t_axis = range(0; step=Δt_d, length=size(ground_truth_energy,2)) ./ 365 # [y]

function moving_average(A::AbstractArray, m::Int)
    N = length(A) 
    out = zeros(eltype(A), N-m)
    for i=1:(N-m)
        out[i] = mean(A[i:i+m])
    end 
    return out 
end

## PLOT 1 QG3 

p = plot() 

if EXPERIMENT == :qg3 

    N_avg = 80
    plot!(p, t_axis[1:end-N_avg],moving_average(ground_truth_energy[1,:],N_avg), color=:black, label="GT 200hPa")
    plot!(p, t_axis[1:end-N_avg],moving_average(ground_truth_energy[2,:],N_avg), color=:black, label="GT 500hPa")
    plot!(p, t_axis[1:end-N_avg],moving_average(ground_truth_energy[3,:],N_avg), color=:black, label="GT 800hPa")


    plot!(p, t_axis[1:end-N_avg],moving_average(psn_energy[1,:],N_avg), color=:blue, label="PSN 200hPa")
    plot!(p, t_axis[1:end-N_avg],moving_average(psn_energy[2,:],N_avg), color=:blue, label="PSN 500hPa")
    plot!(p, t_axis[1:end-N_avg],moving_average(psn_energy[3,:],N_avg), color=:blue, label="PSN 800hPa", title="Average Kinetic Energy", xlabel="Time [y]", ylabel="Kinetic Energy [model units]")

    plot!(p, t_axis[1:size(unet_energy,2)-N_avg],moving_average(unet_energy[1,:],N_avg), color=:red, label="PS UNET 200hPa")
    plot!(p, t_axis[1:size(unet_energy,2)-N_avg],moving_average(unet_energy[2,:],N_avg), color=:red, label="PS UNET 500hPa")
    plot!(p, t_axis[1:size(unet_energy,2)-N_avg],moving_average(unet_energy[3,:],N_avg), color=:red, label="PS UNET 500hPa")

    savefig(p, "qg3-kinetic-energy.pdf")
end 

## PLOT 2 SPEEDY

if EXPERIMENT == :speedy 
    yaxis_mode = :identity
    N_avg = 70

    plot!(p, t_axis[1:end-N_avg],moving_average(ground_truth_energy[1,:],N_avg), color=:black, label="GT 200hPa")
    plot!(p, t_axis[1:end-N_avg],moving_average(ground_truth_energy[2,:],N_avg), color=:black, label="GT 500hPa")
    plot!(p, t_axis[1:end-N_avg],moving_average(ground_truth_energy[3,:],N_avg), color=:black, label="GT 800hPa")


    plot!(p, t_axis[1:size(psn_energy,2)-N_avg],moving_average(psn_energy[1,:],N_avg), color=:blue, label="PSN 200hPa")
    plot!(p, t_axis[1:size(psn_energy,2)-N_avg],moving_average(psn_energy[2,:],N_avg), color=:blue, label="PSN 500hPa")
    plot!(p, t_axis[1:size(psn_energy,2)-N_avg],moving_average(psn_energy[3,:],N_avg), color=:blue, label="PSN 800hPa", title="Average Kinetic Energy", xlabel="Time [y]", ylabel="Kinetic Energy [model units]")

    plot!(p, t_axis[1:size(unet_energy,2)-N_avg],moving_average(unet_energy[1,:],N_avg), color=:red, label="PS UNET 200hPa")
    plot!(p, t_axis[1:size(unet_energy,2)-N_avg],moving_average(unet_energy[2,:],N_avg), color=:red, label="PS UNET 500hPa")
    plot!(p, t_axis[1:size(unet_energy,2)-N_avg],moving_average(unet_energy[3,:],N_avg), ylims=[0,2], color=:red, label="PS UNET 500hPa")

    savefig("speedy-kinetic-energy.png")
end 