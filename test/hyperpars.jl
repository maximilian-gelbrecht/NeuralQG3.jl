@testset "Hyperpars" begin
    h1 = PSNHyperparameters()
    h1 = h1.pars
    @test :N_layers in keys(h1)
    @test :unet_kwargs in keys(h1)
    @test :activation in keys(h1)

    h2 = PSNHyperparameters(N_layers=5, unet_kernel=(2,2))
    h2 = h2.pars
    @test h2[:N_layers] == 5
    @test h2[:unet_kwargs][:kernel] == (2,2)

    th1 = NeuralQG3.training_hyperpars()
    @test :DT_FAC in keys(th1)
    @test :N_batch in keys(th1)
    
    th2 = NeuralQG3.training_hyperpars(N_epoch_ad=50)
    @test th2[:N_epoch_ad] == 50

    names = NeuralQG3.setup_savenames(["test-name"], verbose=false)
    @test names[:SAVE_NAME] == "test-name-model-pars.jld2"
    @test names[:SAVE_NAME_RESULTS] == "test-name-training-results.jld2"

    names = NeuralQG3.setup_savenames(["test-name"], job_id=1, verbose=false)
    @test names[:SAVE_NAME] == "test-name-1-model-pars.jld2"
    @test names[:SAVE_NAME_RESULTS] == "test-name-1-training-results.jld2"

    names = NeuralQG3.setup_savenames(default_name="psn-model", verbose=false)
    @test names[:SAVE_NAME] == "psn-model-model-pars.jld2"
    @test names[:SAVE_NAME_RESULTS] == "psn-model-training-results.jld2"
end 