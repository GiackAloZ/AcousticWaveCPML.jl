module AcousticWaveCPML

    using ParallelStencil

    # Folders for saving documents
    DOCS_FLD = joinpath(dirname(@__DIR__), "simulations")
    TMP_FLD = joinpath(DOCS_FLD, "tmp")

    # Utilites functions
    include("utils.jl")

    # 1D acoustic solver module
    include("Acoustic1D.jl")

    # 2D acoustic solver module
    include("Acoustic2D.jl")

    # xPU 2D acoustic solver module using Threads
    ParallelStencil.@reset_parallel_stencil()
    include("Acoustic2D_Threads.jl")

    # xPU 2D acoustic solver module using CUDA
    ParallelStencil.@reset_parallel_stencil()
    include("Acoustic2D_CUDA.jl")

end # module AcousticWaveCPML
