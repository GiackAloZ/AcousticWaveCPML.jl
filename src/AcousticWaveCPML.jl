module AcousticWaveCPML

    # Folders for saving documents
    DOCS_FLD = joinpath(dirname(@__DIR__), "simulations")
    TMP_FLD = joinpath(DOCS_FLD, "tmp")

    # Utilites functions
    include("utils.jl")

    # 1D acoustic solver module
    include("Acoustic1D.jl")

    # 2D acoustic solver module
    include("Acoustic2D.jl")

end # module AcousticWaveCPML
