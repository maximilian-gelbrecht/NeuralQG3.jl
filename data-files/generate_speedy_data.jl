import Pkg 

Pkg.activate("data-files") # or "." if data-files is the active directory 

using SpeedyWeather, Dates, NeuralQG3, QG3

S, qg3ppars, ψ_0, q_0 = load_data("T42", GPU=false)
qg3p = QG3Model(qg3ppars)

# configuration used for the paper 
trunc = 42
period_spinup = Day(90)
period_data = Day(220)

DT_qg3 = 2π/144*10 
time_unit_qg3 = qg3ppars.time_unit
Δt_data =  Second(Int(ceil(DT_qg3 * time_unit_qg3 * 24*60*60))) 

# generate data 
output_path = "data-files/speedy/" # adjust this in case it isn't already the active directory
output_name = "speedy-qt-precomputed.jld2"
save_name = string(output_path, output_name)

# taken from NeuralQG3.generate_speedy_data 
spectral_grid = SpectralGrid(trunc=trunc, Grid=FullGaussianGrid, nlev=3, vertical_coordinates=SpeedyWeather.SigmaCoordinates([0,0.4,0.6,1]))
output_writer = OutputWriter(spectral_grid, PrimitiveDry, output_dt=Δt_data, output_vars=[:vor], path=output_path)
model = PrimitiveDryModel(; spectral_grid, orography = EarthOrography(spectral_grid), output=output_writer)
simulation = SpeedyWeather.initialize!(model)
run!(simulation, period=period_spinup, output=false)
run!(simulation, period=period_data, output=true)

# taken from NeuralQG3
q = NeuralQG3.process_speedy_data(model, qg3p)
t = NeuralQG3.time_axis_speedy(model, qg3p)

jldsave(save_name; q, t)
