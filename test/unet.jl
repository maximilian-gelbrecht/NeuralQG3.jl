# test U net by learning the \psi to q transform 
using Random, Lux, Optimisers, Zygote, StatsBase, JLD2, ParameterSchedulers, NNlib

@testset "UNet" begin 
    LOAD_DATA = false 

    if LOAD_DATA
        @load "unet-training-data.jld2" training_data 
    else 
        training_data = []
        for it ∈ T(0.):T(0.02):t_end
            sol_it = sol(it)
            sol_grid = transform_grid(sol_it, qg3p)
            y_grid = transform_grid(QG3.qprimetoψ(qg3p, sol_it), qg3p)

            sol_grid = permutedims(sol_grid, (2,3,1))
            y_grid = permutedims(y_grid, (2,3,1))

            push!(training_data, (reshape(sol_grid, size(sol_grid)..., 1), reshape(y_grid, size(y_grid)..., 1)))
        end 
    end

    model = NeuralQG3.UNet(N_channels=[12,24,48,96], activation=NNlib.relu)

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    opt = Optimisers.AdamW(1f-2, (9.0f-1, 9.99f-1), 1.0f-6)
    opt_state = Optimisers.setup(opt, ps) 
    η_schedule = SinExp(λ0=1f-2,λ1=1f-4,period=20,γ=0.97f0)

    function loss(x, y, model, ps, st) 
        ŷ, st = model(x, ps, st)
        return sum(abs2, y - ŷ), st
    end

    loss(training_data[1][1], training_data[1][2], model, ps, st)[1]

    for i_epoch in 1:20
        Optimisers.adjust!(opt_state, η_schedule(i_epoch)) 

        for (x,y) in training_data
            # compute the gradients of the model 
            loss_val, gs = Zygote.withgradient(ps -> loss(x, y, model, ps, st)[1], ps)

            # and update the model with them 
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
        end 
        loss_val_i = mean([loss(data_i[1], data_i[2], model, ps, st)[1] for data_i in training_data])
        println("Epoch [$i_epoch]: Loss $loss_val_i")
    end 

    loss_val_i = mean([loss(data_i[1], data_i[2], model, ps, st)[1] for data_i in training_data])

    @test loss_val_i < 1f-1
    # worked very well Epoch [50]: Loss 0.0019494231, Epoch [100]: Loss 0.0010172721

    # test RecursiveNet
    model2 = NeuralQG3.RecursiveNet(model)
    @test model2(training_data[1][1], ps , st)[1] ≈ model(training_data[1][1], ps, st)[1]

end

