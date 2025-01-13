import Pkg
Pkg.activate("scripts")
using DataFrames, CSV, StatsBase, Plots, NeuralQG3 

__, qg3ppars, __, __ = load_data("T42", GPU=false)

df = DataFrame(CSV.File("results/stability-times.csv"))


times_psn_qg3 = df[!,:PSN_QG3] ./ 365
times_unet_qg3 = df[!,:UNET_QG3] ./ 365

default_val = 100

# split Inf (>50) from the rest 
println(">100y PSN QG3: ",sum(isinf.(times_psn_qg3)))
println(">100y UNET QG3: ",sum(isinf.(times_unet_qg3)))

times_psn_qg3 = times_psn_qg3[@. ~isinf.(times_psn_qg3)]
times_unet_qg3 = times_unet_qg3[@. ~isinf.(times_unet_qg3)]

std_psn_qg3 = std(times_psn_qg3)
std_unet_qg3 = std(times_unet_qg3)

scatter([1,2], [default_val, mean(times_unet_qg3)], ylims=[0, 60], yerror=[0, std_unet_qg3], xlims=[0,4], ylabel="Years [y]",shape=:+, markersize=3, markercolor=:black, tickfontsize=14, aspect_ratio=0.5,linewidth=5, title="Time to Blowup/Instability, Δt≈0.8h")
savefig("results/qg3-long/qg3-stab.pdf")


times_psn_speedy = df[!,:PSN_SPEEDY] ./ 365 
times_unet_speedy = df[!,:UNET_SPEEDY] ./ 365

# split Inf (>50) from the rest 
println(">100y PSN QG3: ",sum(isinf.(times_psn_qg3)))
println(">100y UNET QG3: ",sum(isinf.(times_unet_qg3)))

std_psn_speedy = std(times_psn_speedy)
std_unet_speedy = std(times_unet_speedy)

scatter([1, 2], [5 ,mean(times_unet_speedy)], ylabel="Years [y]",shape=:+, markersize=3, markercolor=:black, yerror=[std_psn_speedy, std_unet_speedy], tickfontsize=14, aspect_ratio=0.5,linewidth=5, title="Time to Blowup/Instability, Δt≈0.8h", ylims=[0,5])
savefig("results/speedy/speedy-stab.pdf")
