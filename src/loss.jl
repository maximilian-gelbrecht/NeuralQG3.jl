struct SpectralFilteringLoss{A,P}
    filter_indices::A
    p::P
    damp_pro_step::Integer
end 

function (l::SpectralFilteringLoss)(x, model, ps, st) 
    ŷ, st = model(x, ps, st) # here ellipsesnotation is used, as a view with selectdim can result in errors on GPU
    sum(abs2, (x[2] - ŷ)[l.filter_indices])
end 

"""
SpectralFilteringLoss(qg3p::QG3ModelParameters, N=5; damp_pro_step::Integer=2)

Iniatiallies a loss function that progressively filters out higher wavenumbers over `N` steps
"""
function SpectralFilteringLoss(p::QG3ModelParameters, N::Integer=5; damp_pro_step::Integer=2, N_batch::Integer=1)

    filters = []

    push!(filters, BitArray(QG3.reorder_SH_gpu(QG3.SH_zero_mask(p), p)))
    L =  QG3.reorder_SH_gpu(QG3.lMatrix(p.L, p.M), p)
    L_i_max = range(start=p.L,step=-damp_pro_step, length=N-1)
    for i=2:N 
        
        Li = BitArray(0 .< L .< L_i_max[i-1])
        Li[1,1] = 1
            
        push!(filters, Li)
    end 
    
    filters = cat(filters..., dims=3)

    filters = reshape(filters, 1, size(filters,1), size(filters,2), 1, size(filters,3))
    filters = repeat(filters, 3, 1, 1, N_batch, 1)
    
    return SpectralFilteringLoss(filters, p, damp_pro_step)
end 
SpectralFilteringLoss(q::QG3Model, varargs...; kwargs...) = SpectralFilteringLoss(q.p, varargs...; kwargs...)

function SpectralFilteringLoss(p::QG3ModelParameters, N_range::AbstractVector; kwargs...)
    [SpectralFilteringLoss(p, i; kwargs...) for i in N_range]
end