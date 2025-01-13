using Dates, NetCDF, SpeedyWeather, CFTime

"""
$(TYPEDSIGNATURES)
Run SpeedyWeather.jl and save its output to a netcdf and its model setup to a JLD2 file 
"""
function generate_speedy_data(; trunc=42, period_spinup=Day(90), period_data=Day(10), Δt_data=10, additional_output=[], kwargs...)

    tmp_output_path = mktempdir(pwd(), prefix = "tmp_outputruns_")  # Cleaned up when the process exits

    spectral_grid = SpectralGrid(trunc=trunc, Grid=FullGaussianGrid, nlev=3, vertical_coordinates=SpeedyWeather.SigmaCoordinates([0,0.4,0.6,1]))
    output_writer = OutputWriter(spectral_grid, PrimitiveDry, output_dt=Δt_data, output_vars=[:vor, additional_output...], path=tmp_output_path)
    model = PrimitiveDryModel(; spectral_grid, orography = EarthOrography(spectral_grid), output=output_writer)
    simulation = SpeedyWeather.initialize!(model)
    run!(simulation, period=period_spinup, output=false)
    run!(simulation, period=period_data, output=true)

    return model
end 

function generate_speedy_data(pars::QG3ModelParameters, DT::Number; kwargs...)
    qgtime_h = QGtime_in_hours(DT, pars)
    speedy_time_step = compute_Δt_for_speedy(qgtime_h)

    return generate_speedy_data(trunc = pars.L - 1, Δt_data=speedy_time_step, kwargs...)
end

"""
    process_speedy_data(m::SpeedyWeather.ModelSetup, p::QG3Model)

Loads an already computed Speedy trajectory from netCDF and converts it into QG potential voriticity q 
"""
function process_speedy_data(m::SpeedyWeather.ModelSetup, p::QG3Model)
    # load it 
    vor = SpeedyWeather.load_trajectory("vor", m)

    # vor -> ψ
    ψ = vor_to_ψ(vor, m) # units might not be correct

    # scaling / units
    ψ ./= p.p.ψ_unit
    
    # ψ -> ψ_SH 
    ψ = permutedims(ψ, (3,2,1,4))

    if QG3.isongpu(p) # for the transforms we need the data on GPU
        ψ = CuArray(ψ)
    end 

    ψ = QG3.transform_SH_data(ψ, p)

    # ψ -> q'
    q = QG3.ψtoqprime(p, ψ)
    
    if QG3.isongpu(p) # we always want to return CPU arrays from this routine
        q = QG3.reorder_SH_cpu(q, p.p)
    end 

    return q 
end



"""
    compute_Δt_for_speedy(dt_hours::T) 

Computes the time step for speedy, it just converts the input into the Dates format properly 
"""
function compute_Δt_for_speedy(dt_hours::T) where T
    return Dates.Second(Int(ceil(dt_hours*60*60)))
end 

function time_axis_speedy(m::SpeedyWeather.ModelSetup, qg3p::QG3Model{T}) where T
    t_speedy = SpeedyWeather.load_trajectory("time", m) # Datetime format 
    t_speedy = t_speedy - t_speedy[1] # millisecond 
    t_speedy = Dates.Second.(t_speedy)

    t = Float64.(Dates.value.(t_speedy))
    t ./= (60*60*24) # days 
    t ./= qg3p.p.time_unit # qg3 time 

    return t 
end 

"""
    vor_to_ψ(vor, S)

SpeedyWeather routine to convert voriticity output (e.g. from a netCDF file) to streamfunction ψ by applying the inverted Laplacian. Returns gridded data
"""
function vor_to_ψ(vor, M::SpeedyWeather.ModelSetup, dtype=Float32)
    S = M.spectral_transform
    
    r = try
        M.geometry.radius # new Speedy structure
    catch e
        M.constants.radius # old Speedy structure
    end 
    
    if sum(ismissing.(vor)) != 0 
        error("There are some missing values in the input!")
    else 
        vor = dtype.(vor)
    end

    ψ = similar(vor)   
    for ilvl ∈ axes(vor,3)
        for it ∈ axes(vor,4)
            vor_spectral = SpeedyWeather.spectral(vor[:,:,ilvl,it])
            ψ_spectral = similar(vor_spectral)
            SpeedyWeather.∇⁻²!(ψ_spectral, vor_spectral, S)
            ψ_spectral .*= (r*r) 
            ψ[:,:,ilvl,it] = Matrix(SpeedyWeather.gridded(ψ_spectral, S))
        end 
    end 
    return ψ
end 

"""
    time_in_SI(time_in_QG3::T, QG3P::QG3ModelParameters{T}) 

Returns the time in SI units (seconds), given the time in the natural units of the QG3.jl model. The natural units of the QG3.jl model are set by assuming time in days and the planetary voriticty factor 2Ω to be equal to one. 
"""
time_in_SI(time_in_QG3::T, QG3P::QG3ModelParameters{T}) where T<:Number = time_in_QG3 * QG3P.time_unit * T(24*60*60)

QGtime_in_hours(time_in_QG3::T, QG3P::QG3ModelParameters{T}) where T = time_in_SI(time_in_QG3, QG3P)/60/60

"""
$(TYPEDSIGNATURES)
Returns a trajectory `(t,q)` for the voriticty computed from the SpeedyWeather `PrimitiveDryModel`.
"""
function compute_speedy_data(qg3p::QG3Model{T}, DT::Number; period_data=Day(300)) where T

    DT_h = NeuralQG3.QGtime_in_hours(DT, qg3p.p)
    dt_speedy = NeuralQG3.compute_Δt_for_speedy(DT_h)
    m = NeuralQG3.generate_speedy_data(period_data=period_data, trunc=42, Δt_data=dt_speedy)

    q = NeuralQG3.process_speedy_data(m, qg3p)
    t = NeuralQG3.time_axis_speedy(m, qg3p)
    m = nothing # free up memory 

    return (T.(t),T.(q))
end

"""
$(TYPEDSIGNATURES)
Returns a trajactory `(t,q)` of the vorticity computed from ERA5 data in units of QG3.jl 
"""
function compute_ERA_data(era5_file::String, qg3p::QG3Model{T}) where T

    ψ = ncread(era5_file,"atmosphere_horizontal_streamfunction")[:,:,:,:]

    lvl = ncread(era5_file,"level")
    times = CFTime.timedecode( ncread(era5_file,"time"),ncgetatt(era5_file,"time","units"))
    times = Dates.value.(Second.(times - times[1])) 

    times = T.((times ./ (60*60*24)) ./ qg3p.p.time_unit) # to qg3.jl units

    LEVELS = [200, 500, 800]

    ψ = ψ[:,:,level_index(LEVELS,lvl),:]
    ψ = permutedims(ψ, [3,2,1,4]) # level, lat, lon,
    ψ = T.(ψ) ./ qg3p.p.ψ_unit

    if QG3.isongpu(qg3p)
        ψ = CuArray(ψ)
    end

    ψ_SH = QG3.transform_SH_data(ψ, qg3p)
    q = ψtoqprime(qg3p, ψ_SH)

    if QG3.isongpu(qg3p)
        q = QG3.reorder_SH_cpu(q, qg3p.p)
    end

    return (times, q)
end


"""
$(TYPEDSIGNATURES)
Load or compute vorticity data from one of the data models `:qg3`, `:speedy` or `:era`. Returns tuple `(t, q(t))`.
"""
function get_data(DATA_MODEL::Symbol, qg3p::QG3Model{T}, COMPUTE_DATA::Bool; S=nothing, DT=nothing, q_0=nothing, t_transient=100, reltol=1e-5, t_save_length=nothing, period_data=Day(2*365), speedy_file_name="data-files/speedy/speedy-qt-precomputed.jld2", era_file_path="/p/projects/ou/labs/ai/reanalysis/era5/T42/streamfunction/hourly/ERA5-sf-200500800hPa-90.nc") where T 


    if COMPUTE_DATA 
        if DATA_MODEL == :qg3

            println("Computing QG3 Data")

            @assert !isnothing(S) "Specifiy Forcing S kwarg"
            @assert !isnothing(DT) "Specifiy DT kwarg"
            @assert !isnothing(q_0) "Specifiy q_0 kwarg"


            @assert !isnothing(t_save_length) "Specifiy t_save_length kwarg"
            (t, q) = NeuralQG3.compute_QG3_data(qg3p, q_0, S, DT; t_save_length = t_save_length, t_transient = t_transient, reltol=reltol)

            @save "qg3-sol.jld2" q t

        elseif DATA_MODEL == :speedy 

            println("Computing Speedy Data")

            @assert !isnothing(DT) "Specifiy DT kwarg"
            @assert !isnothing(period_data) "Period data kwarg"

            (t, q) = NeuralQG3.compute_speedy_data(qg3p, DT; period_data = period_data)
            @save "speedy-qt-long.jld2" q t 

        elseif DATA_MODEL == :era 

            println("Loading ERA Data")

            @assert !isnothing(era_file_path) "Specifiy era_file_path kwarg"
            (t, q) = NeuralQG3.compute_ERA_data(era_file_path, qg3p)
            DT = T(t[2] - t[1])

        else
            error("Unknown DATA_MODEL")
        end

    else 
        if DATA_MODEL == :qg3
            println("Loading QG3 Data")

            @load "qg3-sol.jld2" q t
        elseif DATA_MODEL == :speedy 
            println("Loading Speedy Data")

            @load speedy_file_name q t
        elseif DATA_MODEL == :era 
            println("Loading ERA Data")

            @assert !isnothing(era_file_path) "Specifiy era_file_path kwarg"
            (t, q) = NeuralQG3.compute_ERA_data(era_file_path, qg3p)
            DT = T(t[2] - t[1])

        else 
            error("Unknown DATA_MODEL")
        end
    end

    return (t, q)
end

    