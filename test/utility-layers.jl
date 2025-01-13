using Zygote, StatsBase, Lux, Random

# The tests test the lower level functions, so that they are independend from Lux/Flux
@testset "Utility Layers" begin
    
    # FlattenSH 
    A = rand(2, 10,10)
    mask = zeros(Bool, 2, 10, 10)
    mask[1,1,1] = 1 
    mask[1,5,5] = 1 
    mask[2,1,1] = 1
    mask[2,5,5] = 1

    FA = NeuralQG3.FlattenSH(mask)
    FA_array, __ = FA(A, nothing, nothing)

    @test FA_array ≈ [A[1,1,1] A[1,5,5]; A[2,1,1] A[2,5,5]]

    # ExpandSH
    
    EA = NeuralQG3.ExpandSH(mask) 
    EA_array, __ = EA(FA_array, nothing, nothing)

    @test EA_array[1,1,1] ≈ A[1,1,1]
    @test EA_array[1,5,5] ≈ A[1,5,5]
    @test EA_array[2,1,1] ≈ A[2,1,1]
    @test EA_array[2,5,5] ≈ A[2,5,5]

    # AD ExpandSH 
    # a simple test model x -> FlattenSH -> sin(x) -> ExpandSH
    f(x) = EA(sin.(FA(x, nothing, nothing)[1]), nothing, nothing)[1]
    y, back = Zygote.pullback(f, A)
    pb = back(ones(size(A)))

    @test pb[1][1,1,1] ≈ cos(A[1,1,1])
    @test pb[1][1,5,5] ≈ cos(A[1,5,5])
    @test pb[1][2,1,1] ≈ cos(A[2,1,1])
    @test pb[1][2,5,5] ≈ cos(A[2,5,5])

    # SPHZNormalize
    __, qg3ppars, __, q_0 = QG3.load_precomputed_data()

    sph = NeuralQG3.SPHZNormalize(qg3ppars)
    mask = QG3.reorder_SH_gpu(QG3.SH_zero_mask(qg3ppars, (1,1,1)),  qg3ppars)
    N_nonzero = sum(mask)
    N_zero = prod(size(mask)) - N_nonzero
    μ, σ = NeuralQG3.sph_z_normaliziation(q_0, N_nonzero, N_zero)

    qn = ((q_0 .- μ)./σ) .* mask

    qn_flat = NeuralQG3.flatten_SH(qn, QG3.reorder_SH_gpu(QG3.SH_zero_mask(qg3ppars, (1,1)), qg3ppars))
    
    @test sum(abs.(mean(qn_flat, dims=2))) < 1e-5
    @test abs(sum(std(qn_flat, dims=2)) - 3) < 1e-2

    # ScaledInitDense 
    d = Dense(10,10)

    scaled_d = NeuralQG3.ScaledInitDense(d, 0.01)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, d)
    ps2, st2 = Lux.setup(rng, scaled_d)

    @test std(ps[:weight]) / std(ps2[:weight]) > 50f0

end 