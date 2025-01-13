using Random, Lux, Optimisers, Zygote, ParameterSchedulers, StatsBase


# as a test we will let the pseudospectralnet learn the T_ψq operator 
@testset "PseudoConv" begin 

    training_data = []
    for it ∈ T(0.):T(0.02):t_end
        push!(training_data, (sol(it), QG3.qprimetoψ(qg3p, sol(it))))
    end 

    model = NeuralQG3.default_shconvnet(qg3p, 3=>3)

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    opt = Optimisers.AdamW(1f-2, (9.0f-1, 9.99f-1), 1.0f-6)
    opt_state = Optimisers.setup(opt, ps) 
    η_schedule = SinExp(λ0=1f-2,λ1=1f-4,period=50,γ=0.99f0)

    function loss(x, y, model, ps, st) 
        ŷ, st = model(x, ps, st)
        return sum(abs2, y - ŷ), st
    end

    loss(training_data[1][1], training_data[1][2], model, ps, st)[1]

    for i_epoch in 1:400
        Optimisers.adjust!(opt_state, η_schedule(i_epoch)) 

        for (x,y) in training_data
            # compute the gradients of the model 
            gs = Zygote.gradient(ps -> loss(x, y, model, ps, st)[1], ps)

            # and update the model with them 
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
        end 

        loss_val_i = mean([loss(training_data[i][1], training_data[i][2], model, ps, st)[1] for i=1:length(training_data)])
        #println("Epoch [$i_epoch]: Loss $loss_val_i")
    end 

    loss_val_i = mean([loss(training_data[i][1], training_data[i][2], model, ps, st)[1] for i=1:length(training_data)])

    @test loss_val_i < 0.1f0 

end 