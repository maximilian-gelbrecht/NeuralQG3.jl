"""
    load_data(name::String, path="data-files/", GPU::Bool)

Loads pre-computed data for the QG Model, e.g. for a `name=T42` grid.  
"""
function load_data(name::String, path="/../data-files/"; GPU::Bool=false)

    root_path = @__DIR__
    
    if name=="T60"
        @load string(root_path, path, "t60-precomputed-S.jld2") S
        @load string(root_path, path, "t60-precomputed-p.jld2") qg3ppars
        @load string(root_path, path, "t60-precomputed-sf.jld2") ψ_0
        @load string(root_path, path, "t60-precomputed-q.jld2") q_0
    elseif name=="T42"
        @load string(root_path, path, "t42-precomputed-S.jld2") S
        @load string(root_path, path, "t42-precomputed-p.jld2") qg3ppars
        @load string(root_path, path, "t42-precomputed-sf.jld2") ψ_0
        @load string(root_path, path, "t42-precomputed-q.jld2") q_0
    elseif name=="T21"
        S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()
    else
        error("Unknown grid, only T21, T42 and T60 available")
    end

    if GPU
        S, qg3ppars, ψ_0, q_0 = QG3.reorder_SH_gpu(S, qg3ppars), togpu(qg3ppars), QG3.reorder_SH_gpu(ψ_0, qg3ppars), QG3.reorder_SH_gpu(q_0, qg3ppars)
    end
    return S, qg3ppars, ψ_0, q_0
end

"""
    animation(trajectory; title="Test", ΔN_frames=1, kwargs...)

Animates the trajectory every `ΔN_frames` timestep. Output the animation with `Plots.gif(anim, name, fps = 30)`
"""
function animation(trajectory; title="Test", ΔN_frames=1, clims=nothing, return_clims=false, kwargs...)
    t, x = trajectory 

    if isnothing(clims)
        clims = (-1.1*maximum(abs.(x)),1.1*maximum(abs.(x))) # get colormap maxima
    end 

    anim = @animate for it ∈ 1:ΔN_frames:length(t)
        heatmap(x[:,:,it], c=:balance, clims=clims, title=string(title, " at i_t=",it), kwargs...)
    end 

    if return_clims
        return anim, clims 
    else 
        return anim 
    end
end 

"""
$(TYPEDSIGNATURES)
Compute training data from the QG3 Model.
"""
function compute_QG3_data(qg3p::QG3Model{T}, q_0, S, DT::Number; t_save_length::Number, t_transient=T(100), reltol=1e-5) where T
    DT = T(DT)
    t_save_length = T(t_save_length)
    t_end = t_transient + t_save_length

    prob = ODEProblem(QG3.QG3MM_gpu,q_0,(T(0.),t_end),[qg3p, S])
    sol = @time solve(prob, Tsit5(), dt=DT, saveat=t_transient:DT:t_end, reltol=reltol)
        
    q = QG3.reorder_SH_cpu(Array(sol), qg3p.p) # cpu for saving 
    t = sol.t 

    return (t, q)
end 
