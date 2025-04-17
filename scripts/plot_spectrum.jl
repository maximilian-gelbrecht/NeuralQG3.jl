import Pkg 
Pkg.activate("scripts")

using Plots, JLD2, StatsBase

BASE_NAME = "results/qg3-long/Spectrum-Year-GT--m12-"
S1 = jldopen(string(BASE_NAME,"1.jld2"))

i_lvl = "S_1"

fig = plot(S1[i_lvl], yscale=:log10)
for i=2:6
    S1 = jldopen(string(BASE_NAME,i,".jld2"))

    plot!(fig, S1[i_lvl], yscale=:log10,label=string(i))


end 

