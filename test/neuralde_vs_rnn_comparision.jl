# in this script we test two different NeuralDE implementations vs the RNN baseline 
# we want to achieve that the ADEulerStep is as fast as the RNN and delivers the same results as the Euler from SciMLSensitivity
# we do this test for the pure UNET

using Zygote, Lux, ComponentArrays, SciMLSensitivity, Random

@testset "ADEulerstep vs SciML vs RNN Baseline" begin 

    test_data = (sol.t[1:2], transform_grid(Array(sol)[:,:,:,1:2], qg3p))
    RELTOL_PREDICT = 1f-3

    nn = Chain(NeuralQG3.PermutedimLayer((2,3,1,4)), NeuralQG3.UNet(), NeuralQG3.PermutedimLayer((3,1,2,4)))

    nn_adeuler = Chain(WrappedFunction(x->reshape(x,size(x)...,1)), nn, WrappedFunction(x->view(x,:,:,:,1)))
    neural_de_adeuler = NeuralDELux.ADNeuralDE(nn_adeuler, alg=NeuralDELux.ADEulerStep(), dt=DT)
    neural_de_sensitivity = NeuralDELux.SciMLNeuralDE(nn_adeuler, alg=Euler(), sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), dt=DT, reltol=RELTOL_PREDICT)

    rnn_model = NeuralQG3.RecursiveNet(nn)

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, neural_de_adeuler)
    ps = ComponentArray(ps) |> gpu

    loss = NeuralDELux.least_square_loss_ad 
    loss_sciml = NeuralDELux.least_square_loss_sciml

    function loss_rnn(x, model, ps, st) 
        ŷ, st = model(x, ps, st)
        return sum(abs2, view(x[2],:,:,:,2:2) - ŷ)
    end

    ps_rnn, st_rnn = Lux.setup(rng, rnn_model)
    ps_rnn = ComponentArray(layer_1=ps[:layer_2], layer_2=ps[:layer_3], layer_3=ps[:layer_4]) |> gpu;

    time_adeuler = @belapsed $neural_de_adeuler($test_data, $ps, $st)[1]
    time_sciml = @belapsed $neural_de_sensitivity($test_data, $ps, $st)[1]
    time_rnn = @belapsed $rnn_model($test_data, $ps_rnn, $st_rnn)[1]

    @test isapprox(time_rnn, time_adeuler, rtol=0.15)
    @test time_sciml > 1.8*time_adeuler

    dt = DT
    res_nn = (rnn_model(test_data, ps_rnn, st_rnn)[1] * dt) + test_data[2][:,:,:,1]
    res_euler = neural_de_adeuler(test_data, ps, st)[1][:,:,:,2]
    res_sciml = neural_de_sensitivity(test_data, ps, st)[1][:,:,:,2];

    @test isapprox(res_nn, res_euler, rtol=1e-5)
    @test isapprox(res_euler, res_sciml, rtol=1e-3) 

    loss_val, gs = Zygote.withgradient(ps -> loss(test_data, neural_de_sensitivity, ps, st)[1], ps)
    loss_val5, gs5 = Zygote.withgradient(ps -> loss_sciml(test_data, neural_de_adeuler, ps, st)[1], ps)
    loss_val6, gs6 = Zygote.withgradient(ps -> loss_rnn(test_data, rnn_model, ps_rnn, st_rnn)[1], ps)

    @test isapprox(gs5[1][:layer_3] , gs[1][:layer_3],rtol=0.05)

    time_sensitivty = @belapsed Zygote.withgradient(ps -> $loss($test_data, $neural_de_sensitivity, ps, $st)[1], $ps)
    time_adeuler = @belapsed Zygote.withgradient(ps -> $loss($test_data, $neural_de_adeuler, ps, $st)[1], $ps)
    time_rnn = @belapsed Zygote.withgradient(ps -> $loss_rnn($test_data, $rnn_model, ps, $st_rnn)[1], $ps_rnn)

    @test isapprox(time_adeuler, time_rnn, rtol=0.1)
    @test time_sensitivty > 2*time_adeuler
end