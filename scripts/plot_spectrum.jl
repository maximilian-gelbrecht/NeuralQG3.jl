import Pkg 
Pkg.activate("scripts")

using Plots, JLD2, StatsBase, Statistics

#BASE_NAME = "results/qg3-long/Spectrum-Baseline-qg3-ev0-m36-"

BASE_NAME = "results/qg3-long/Spectrum-PSN-qg3-ev0-m36-"
#BASE_NAME = "results/speedy/Spectrum-PSN-speedy-ev3-m36-"
#BASE_NAME = "results/speedy/Spectrum-Baseline-speedy-ev3-m36-"

S1 = jldopen(string(BASE_NAME,"1.jld2"))

function plot_angular_power(BASE_NAME, i_lvl, title="")

    fig = plot(S1[i_lvl], yscale=:log10, xlabel="Wavenumber l", ylabel="Average Angular Power", title=title)
    for i=1:20
        S1 = jldopen(string(BASE_NAME,i,".jld2"))

        plot!(fig, S1[i_lvl], yscale=:log10,label=string(i), legend=:none)


    end 
    return fig
end     

p1 = plot_angular_power(BASE_NAME, "S_1", "PSN 200 hPa")
p2 = plot_angular_power(BASE_NAME, "S_2", "PSN 500 hPa")
p3 = plot_angular_power(BASE_NAME, "S_3", "PSN 850 hPa")

plot(p1, p2, p3, layout=(1,3), size=(1100,300))

BASE_NAME = "results/qg3-long/Spectrum-Baseline-qg3-ev0-m36-"

p4 = plot_angular_power(BASE_NAME, "S_1", "PS UNET 200 hPa")
p5 = plot_angular_power(BASE_NAME, "S_2", "PS UNET 500 hPa")
p6 = plot_angular_power(BASE_NAME, "S_3", "PS UNET 850 hPa")

plot(p1, p2, p3, p4, p5, p6, layout=(2,3), size=(800,800))

savefig("spectra-qg3.pdf")