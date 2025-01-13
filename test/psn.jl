# for the full PSN we dont test the full training, only that the gradient compiles and returns a value 
using QG3, NeuralQG3, JLD2, OrdinaryDiffEq, Optimisers, Random, SciMLSensitivity, Zygote, Statistics, CUDA, NNlib, Lux, ComponentArrays

@testset "PSN" begin

    N_Nodes = 30 
    N_layers = 2
    N_channels = 3 
    func = selu 

    rng = Random.default_rng()
    x = cat(q_0, ψ_0, dims=1)
    x_batched = repeat(x,1,1,1,2)
    ψ_0_grid = transform_grid(ψ_0, qg3p)
    ψ_0_grid_batched = repeat(ψ_0_grid, 1,1,1,2)
    N_batch = 2

    # KnowledgeGridLayer 
    knowledge_layer = NeuralQG3.KnowledgeGridLayer(qg3p; additional_knowledge=false)

    out = knowledge_layer(x, nothing, nothing)[1]
    @test size(out,2) == size(ψ_0_grid,2) # output in grid space
    @test size(out,3) == size(ψ_0_grid,3)

    knowledge_layer = NeuralQG3.KnowledgeGridLayer(qg3p; additional_knowledge=false, N_batch=N_batch)

    out = knowledge_layer(x_batched, nothing, nothing)[1]
    @test size(out,2) == size(ψ_0_grid_batched,2) # output in grid space
    @test size(out,3) == size(ψ_0_grid_batched,3)
    @test size(out,4) == size(ψ_0_grid_batched,4)

    knowledge_layer = NeuralQG3.KnowledgeGridLayer(qg3p; additional_knowledge=true)

    out = knowledge_layer(x, nothing, nothing)[1]
    @test size(out,2) == size(ψ_0_grid,2) # output in grid space
    @test size(out,3) == size(ψ_0_grid,3)

    knowledge_layer = NeuralQG3.KnowledgeGridLayer(qg3p; additional_knowledge=true, N_batch=N_batch)

    out = knowledge_layer(x_batched, nothing, nothing)[1]
    @test size(out,2) == size(ψ_0_grid_batched,2) # output in grid space
    @test size(out,3) == size(ψ_0_grid_batched,3)
    @test size(out,4) == size(ψ_0_grid_batched,4)

    # KnowledgeSHLayer
    knowledge_layer = NeuralQG3.KnowledgeSHLayer(qg3p; additional_knowledge=false)

    out = knowledge_layer(x, nothing, nothing)[1]
    @test size(out,2) == size(x,2) # output in SH space
    @test size(out,3) == size(x,3)

    knowledge_layer = NeuralQG3.KnowledgeSHLayer(qg3p; additional_knowledge=false, N_batch=N_batch)

    out = knowledge_layer(x_batched, nothing, nothing)[1]
    @test size(out,2) == size(x_batched,2) # output in SH space
    @test size(out,3) == size(x_batched,3)
    @test size(out,4) == size(x_batched,4)

    knowledge_layer = NeuralQG3.KnowledgeSHLayer(qg3p; additional_knowledge=true, S=S)

    out = knowledge_layer(x, nothing, nothing)[1]
    @test size(out,2) == size(x,2)  # output in SH space
    @test size(out,3) == size(x,3)

    knowledge_layer = NeuralQG3.KnowledgeSHLayer(qg3p; additional_knowledge=true, N_batch=N_batch, S=S)

    out = knowledge_layer(x_batched, nothing, nothing)[1]
    @test size(out,2) == size(x_batched,2) # output in SH space
    @test size(out,3) == size(x_batched,3)
    @test size(out,4) == size(x_batched,4)


    # Full PSN 
    # no add knowledge 
    model = PseudoSpectralNet(qg3p, N_layers=N_layers, N_Nodes=N_Nodes, N_channels=N_channels, activation=func, GPU=GPU, additional_knowledge=false, conv_mode=:pseudoconv, znorm=true)

    ps, st = Lux.setup(rng, model)
    y = model(x, ps, st)
    @test size(q_0) == size(y[1]) # test if there is an output and if it is has the correct size

    # add knowledge 
    model = PseudoSpectralNet(qg3p, N_layers=N_layers, N_Nodes=N_Nodes, N_channels=N_channels, activation=func, GPU=GPU, additional_knowledge=true, conv_mode=:pseudoconv, S=S)

    ps, st = Lux.setup(rng, model)
    y = model(x, ps, st)
    @test size(q_0) == size(y[1]) # test if there is an output and if it is has the correct size

    # UNet 
    # no add knowledge
    model = PseudoSpectralNet(qg3p, N_layers=N_layers, N_Nodes=N_Nodes, N_channels=N_channels, activation=func, GPU=GPU, additional_knowledge=false, conv_mode=:unet)

    ps, st = Lux.setup(rng, model)
    y = model(x, ps, st)
    @test size(q_0) == size(y[1]) # test if there is an output and if it is has the correct size

    # UNet, no add knowledge, batches 
    model = PseudoSpectralNet(qg3p, N_layers=N_layers, N_Nodes=N_Nodes, N_channels=N_channels, N_batch=N_batch, activation=func, GPU=GPU, additional_knowledge=false, conv_mode=:unet)

    ps, st = Lux.setup(rng, model)
    y = model(x_batched, ps, st)
    @test size(q_0,1) == size(y[1],1) # test if there is an output and if it is has the correct size
    @test size(q_0,2) == size(y[1],2) # test if there is an output and if it is has the correct size
    @test size(q_0,3) == size(y[1],3) # test if there is an output and if it is has the correct size
    @test size(x_batched,4) == size(y[1],4) # test if there is an output and if it is has the correct size

    # add knowledge
    model = PseudoSpectralNet(qg3p, N_layers=N_layers, N_Nodes=N_Nodes, N_channels=N_channels, activation=func, GPU=GPU, additional_knowledge=true, conv_mode=:unet, S=S)

    ps, st = Lux.setup(rng, model)
    y = model(x, ps, st)
    @test size(q_0) == size(y[1]) # test if there is an output and if it is has the correct size

    rhs = NeuralQG3.PSN_RHS(model, qg3p, NeuralQG3.DetermineDevice(gpu=false))
    neural_de = NeuralDELux.ADNeuralDE(model=rhs, dt=DT)
    ps, st = Lux.setup(rng, neural_de)
    ps = ComponentArray(ps) 

    train_example = (sol.t[1:2],Array(sol(sol.t[1:2])))
    y = neural_de(train_example[2][:,:,:,1], ps, st)
    @test size(q_0) == size(y[1]) # test if there is an output and if it is has the correct size

    # gradient and loss test 
    loss_ad = NeuralDELux.least_square_loss_ad
    loss_val = loss_ad(train_example, neural_de, ps, st)[1]

    # test if loss is really finite scalar 
    @test typeof(loss_val) <: Number 
    @test isfinite(loss_val)

    # gradient test 
    loss_val2, gs = Zygote.withgradient(ps -> loss_ad(train_example, neural_de, ps, st)[1], ps)
    @test loss_val ≈ loss_val2 
    @test !(isnothing(gs[1]))

    #gridforecast test
    valid_trajectory = (sol.t, Array(sol))
    grid_forecast = NeuralQG3.GridForecast(valid_trajectory, qg3p.g.SHtoG, trajectory_call=true, modes=("forecast_delta", "forecast_length", "latitude_delta", "average_forecast_length", "average_forecast_delta"), N_forecast=3, N_avg=3)
    f = grid_forecast(neural_de, ps, st)

    @test !(isnothing(f))

    # hyperpar init 
    h1 = PSNHyperparameters()
    model = PseudoSpectralNet(qg3p, h1)

    ps, st = Lux.setup(rng, model)
    y = model(x, ps, st)
    @test size(q_0) == size(y[1]) # test if there is an output and if it is has the correct size


end

