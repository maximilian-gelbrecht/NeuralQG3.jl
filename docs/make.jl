using NeuralQG3
using Documenter

DocMeta.setdocmeta!(NeuralQG3, :DocTestSetup, :(using NeuralQG3); recursive=true)

makedocs(;
    modules=[NeuralQG3],
    authors="Maximilian Gelbrecht <maximilian.gelbrecht@posteo.de> and contributors",
    sitename="NeuralQG3.jl",
    format=Documenter.HTML(;
        canonical="https://maximilian-gelbrecht.github.io/NeuralQG3.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/maximilian-gelbrecht/NeuralQG3.jl",
    devbranch="main",
)
