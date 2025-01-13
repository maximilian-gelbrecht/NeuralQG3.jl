# this is a basic run from netcdf files, you have to make sure that the data is there. it is not present in the repository where you'll only find example initial conditions and a pre-computed forcing in order to save space in the repository / package.
using Pkg
Pkg.activate("scripts")

using QG3, NetCDF, CUDA, CFTime, NeuralQG3, Dates, BenchmarkTools, OrdinaryDiffEq, JLD2

# first we import the data (streamfunction), land sea mask, orography etc
T = Float32

begin
        #DIR = "/p/tmp/maxgelbr/data/ERA5-uv/"
        DIR = "data/"
        #NAME = "ERA5-sf-t42.nc"
        NAME = "ERA-sf-t42.nc"

        LSNAME = "land-t42.nc"
        ORONAME = "oro-t42.nc"

        LATNAME = "lat"
        LONNAME = "lon"

        lats = deg2rad.(T.(ncread(string(DIR,NAME),LATNAME)))
        lat_inds = 1:size(lats,1)

        ψ = ncread(string(DIR,NAME),"atmosphere_horizontal_streamfunction")[:,:,:,:]

        lvl = ncread(string(DIR,NAME),"level")
        lats = deg2rad.(T.(ncread(string(DIR,NAME),LATNAME)))[lat_inds]
        lons = deg2rad.(T.(ncread(string(DIR,NAME),LONNAME)))

        times = CFTime.timedecode( ncread(string(DIR,NAME),"time"),ncgetatt(string(DIR,NAME),"time","units"))

        summer_ind = [month(t) ∈ [6,7,8] for t ∈ times]
        winter_ind = [month(t) ∈ [12,1,2] for t ∈ times]

        LS = T.(permutedims(ncread(string(DIR,LSNAME),"var172")[:,:,1],[2,1]))[lat_inds,:]
        # Land see mask, on the same grid as lats and lons

        h = (T.(permutedims(ncread(string(DIR,ORONAME),"z")[:,:,1],[2,1]))[lat_inds,:] .* T.(ncgetatt(string(DIR,ORONAME), "z", "scale_factor"))) .+ T.(ncgetatt(string(DIR,ORONAME),"z","add_offset"))
        # orography, array on the same grid as lats and lons

        LEVELS = [200, 500, 800]

        ψ = togpu(ψ[:,:,level_index(LEVELS,lvl),:])
        ψ = permutedims(ψ, [3,2,1,4]) # level, lat, lon,
        ψ = T.(ψ[:,lat_inds,:,:])

        gridtype="gaussian"
end

L = 43 # T21 grid, truncate with l_max = 21

# pre-compute the model and normalize the data
qg3ppars = QG3ModelParameters(L, lats, lons, LS, h)

ψ = togpu(ψ ./ qg3ppars.ψ_unit)

qg3p = CUDA.@allowscalar QG3Model(qg3ppars)

# stream function data in spherical domain
ψ_SH = transform_SH(ψ, qg3p)

# initial conditions for streamfunction and vorticity
ψ_0 = CUDA.@allowscalar ψ_SH[:,:,:,1]
q_0 = QG3.ψtoqprime(qg3p, ψ_0)

# compute the forcing from winter data
S = @time QG3.compute_S_Roads(ψ_SH, qg3p)

# compute a transient and save those ICs 

t_transient = T(2000)
t_end = T(3000)
DT =  T((2π/144)*10)
prob = ODEProblem(QG3.QG3MM_gpu,q_0,(T(0.),t_end),[qg3p, S])

sol = @time solve(prob, Tsit5(), dt=DT, saveat=t_transient:DT:t_end)

q_0 = sol(t_end)
ψ_0 = qprimetoψ(qg3p, q_0)

path = "data-files/"
@save string(path,"t42-precomputed-S.jld2") S
@save string(path,"t42-precomputed-p.jld2") qg3ppars
@save string(path,"t42-precomputed-sf.jld2") ψ_0
@save string(path,"t42-precomputed-q.jld2") q_0

PLOT = true
if PLOT 
    using Plots
    # PSN 
    plot_field = (sol.t, Array(transform_grid(qprimetoψ(qg3p, CuArray(sol)), qg3p))[1,:,:,:])
    anim, clims = NeuralQG3.animation(plot_field, title="QG3 Lvl1", return_clims=true)
    gif(anim, "qg3t42-precompute.gif", fps=30)
end
