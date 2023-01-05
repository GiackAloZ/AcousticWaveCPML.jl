using Documenter, AcousticWaveCPML

makedocs(
    sitename = "AcousticWaveCPML.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "github.com/GiackAloZ/AcousticWaveCPML.jl.git",
    devbranch = "main",
    versions = nothing
)